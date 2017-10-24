__author__ = "Nikhil Mehta"
__copyright__ = "--"

import numpy as np
import torch as t
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import math

# Our Encoder and Generator Class
from .encoder import Encoder
from .generator import Generator
from selfModules.embedding import Embedding
from utils.beam_search import Beam
from .discriminator_sentiment import Sentiment_CNN

from utils.functional import kld_coef

import sys

class Controlled_Generation_Sentence(nn.Module):

    def __init__(self, config, embedding_path):

        super(Controlled_Generation_Sentence, self).__init__()

        self.config = config
        self.embedding = Embedding(self.config, embedding_path)
        self.embedding_param_autograd = [1 if param.requires_grad else 0 for param in self.embedding.parameters()]
        
        self.encoder = Encoder(self.config)
        self.encoder_param_autograd = [1 if param.requires_grad else 0 for param in self.encoder.parameters()]
        
        self.e2mu = nn.Linear(self.config.encoder_rnn_size*2, self.config.latent_variable_size)
        self.e2logvar = nn.Linear(self.config.encoder_rnn_size*2, self.config.latent_variable_size)

        self.generator = Generator(self.config)
        self.generator_param_autograd = [1 if param.requires_grad else 0 for param in self.generator.parameters()]

        self.sentiment_discriminator = Sentiment_CNN(self.config)
        self.sentiment_discriminator_autograd = [1 if param.requires_grad else 0 for param in self.sentiment_discriminator.parameters()]


    def compute_mahalabonis(self, mu1, mu2, std1, std2):
        
        std = (std1 + std2) / 2 
        delta_mu = mu1 - mu2

        #det_std1_sqr = t.sum(std1 * std1, 1)
        #det_std2_sqr = t.sum(std2 * std2, 1)
        #det_std_sqr = t.sum(std*std, 1)

        #print det_std_sqr.size()
        #print det_std1_sqr.size()
        
        distance = t.sum(delta_mu * t.reciprocal(std) * delta_mu, 1)
        #distance = 1/8*distance
        #distance += 1/4 * t.log(det_std_sqr) - 1/8 * t.log(det_std1_sqr * det_std2_sqr) 
        return distance
    
    def encode_for_z (self, encoder_input, batch_size, use_cuda):
        """
        Given encoder_input computes the z
        This is the encoding step.
        """
        context, h_0, c_0 = self.encoder(encoder_input, None)
        State = (h_0, c_0)

        mu = self.e2mu(context)
        logvar = self.e2logvar(context)

        std = t.exp(0.5*logvar)

        z = Variable(t.randn([batch_size, self.config.latent_variable_size]))

        if use_cuda:
            z = z.cuda()

        z = z * std + mu

        return z, std, logvar, mu
    
    def train_rvae (self, drop_prob, encoder_word_input=None, encoder_char_input=None, generator_word_input=None):

        """
        VAE Forward step with random c codeword drawn from bernoulli
        """
        
        use_cuda = self.embedding.word_embed.weight.is_cuda
        
        [batch_size, _] = encoder_word_input.size()
        encoder_input = self.embedding(encoder_word_input, encoder_char_input)

        z, std, logvar, mu = self.encode_for_z(encoder_input, batch_size, use_cuda)

        # use random codeword
        init_prob = t.ones(batch_size, 1)*0.5
        c = Variable(t.bernoulli(init_prob), requires_grad=False)

        if use_cuda:
            c = c.cuda()
        
        input_code = t.cat((z,c), 1)
                    
        kld = (-0.5 * t.sum(logvar - t.pow(mu,2) - t.exp(logvar) + 1, 1)).mean().squeeze()

        generator_input = self.embedding.word_embed(generator_word_input)
        out, final_state = self.generator(generator_input, input_code, drop_prob, None)

        return out, final_state, kld, mu, std, z

    def discriminator_forward_function (self, data_handler, use_cuda):
        """
        Discriminator forward function
        
        Parameters:
        * data_handler: gen_batch_loader from data_handler
        * use_cuda: cuda flag
        * batch_size: batch size for forward prop. This can be higher than generator batch.

        Returns:
        * Discriminator forward step function
        """
        
        self.sentiment_discriminator.eval()
        
        def discriminator_forward(start_index, batch_size):

            """
            This function is used at the start of generator-encoder training interation. 
            The returned codeword can be used as the target value when doing generator
            training.

            Parameters:
            * batch_index: batch index 
            """
            
            input = data_handler.gen_batch_loader.next_batch(batch_size, 'train', start_index)
            [encoder_word_input, encoder_character_input, decoder_word_input, _, target] = input

            encoder_word_input = t.from_numpy(encoder_word_input)
            encoder_word_input = Variable(encoder_word_input)

            if use_cuda:
                encoder_word_input = encoder_word_input.cuda()

            encoder_word_input = self.embedding.get_word_embed(encoder_word_input)
            
            # sentence_hot_input = data_handler.feature_from_indices(encoder_word_input)
            # sentence_hot_input = Variable(sentence_hot_input)
                                                
            logit = self.sentiment_discriminator(encoder_word_input)

            softmax = F.softmax(logit)
            
            # bernoulli here with softmax[:,1]
            c = t.bernoulli(softmax[:, 1])
                        
            return c.data
            
        return discriminator_forward


    def discriminator_sentiment_valid (self, data_handler, use_cuda):

        """
        Initializes the step function for discriminator validation
        
        Parameters:
        * data_handler: data handler for discriminator labeled data
        * use_cuda: cuda flag

        Returns:
        * Discriminator Valid step function
        """
                               
        def valid (batch_index):

            # Handles dropouts at validation
            self.sentiment_discriminator.eval()
            
            # Get labeled training data from the dataset 
            batch_train_X, batch_train_Y = data_handler.get_sentiment_dev_batch(batch_index)
            
            batch_train_X = Variable(data_handler.create_sentiment_batch(batch_train_X))
            batch_train_Y = t.from_numpy(np.array(batch_train_Y))
            batch_train_Y = Variable(batch_train_Y.long())
                                    
            #--------------
            # Labeled Data

            batch_size = batch_train_X.size(0)
                        
            # Torch tensors
            if use_cuda:
                batch_train_X = batch_train_X.cuda()
                batch_train_Y = batch_train_Y.cuda()

            batch_train_X = self.embedding.word_embed(batch_train_X)
            logit = self.sentiment_discriminator(batch_train_X)

            batch_train_ce_loss = F.cross_entropy(logit, batch_train_Y)
            
            return batch_train_ce_loss

        return valid
    
    def discriminator_sentiment_trainer (self, data_handler, use_cuda):

        """
        Initializes the step function for discriminator training
        
        Parameters:
        * data_handler: data handler for discriminator labeled data
        * gen_samples: generated samples from current state of the generator
        * gen_c: codeword used to generate gen_samples
        * use_cuda: cuda flag

        Returns:
        * Discriminator Training step function
        """
        
        # Switch off the requires_grad flag for generator and encoder 
        self.encoder_mode(train_mode=False)
        self.generator_mode(train_mode=False)
        self.discriminator_mode(train_mode=True)

        self.encoder.eval()
        self.generator.eval()
        self.sentiment_discriminator.train()
                
        # get the relevant model parameters
        optimizer = t.optim.Adam(self.sentiment_discriminator_parameters(), self.config.learning_rate)
               
        def train (generated_samples, generated_c, batch_index):

            # This should give samples from both the generator and discriminator dataset
            # These should be torch tensors as one-hot vectors (discriminator dataset) or softmax outputs (Generated sentences)
            # expected dimenstion is (batch_size, seq_len, vocab_size)

            # Get labeled training data from the dataset 
            batch_train_X, batch_train_Y = data_handler.get_sentiment_train_batch(batch_index)

            #-------------- Calculate Loss for labeled data first

            # We will do separate forward pass for generated data and data from the labeled dataset
            # This is simply due to that fact that we have to weight the loss from generted data
            batch_train_X = Variable(data_handler.create_sentiment_batch(batch_train_X))
            batch_generated_X = Variable(data_handler.create_sentiment_batch(generated_samples))

            #batch_train_Y = np.array(batch_train_Y)
            #batch_train_Y = batch_train_Y.reshape(batch_train_Y.shape[0], 1)
            batch_train_Y = t.from_numpy(np.array(batch_train_Y))
            batch_train_Y = Variable(batch_train_Y.long())
            batch_generated_Y = generated_c.view(-1)
            batch_generated_Y = batch_generated_Y.long()
            batch_generated_Y = Variable(batch_generated_Y)
            
            '''
            batch_train_X.extend(generated_samples)

            batch_train_X = data_handler.create_sentiment_batch(batch_train_X)
            batch_train_X = Variable(batch_train_X)
            batch_train_Y = t.from_numpy(np.array(batch_train_Y))
            #batch_train_Y = batch_train_Y.view(batch_train_Y.size(0), 1)

            generated_c = generated_c.view(-1)
            batch_train_Y = batch_train_Y.long()
            generated_c = generated_c.long()
            batch_train_Y = t.cat((batch_train_Y, generated_c) ,dim=0)
            batch_train_Y = Variable(batch_train_Y)
            '''
            #--------------
            # Labeled Data

            batch_size = batch_train_X.size(0)
                        
            # Torch tensors
            if use_cuda:
                batch_train_X = batch_train_X.cuda()
                batch_train_Y = batch_train_Y.cuda()

            batch_train_X = self.embedding.word_embed(batch_train_X)
            logit = self.sentiment_discriminator(batch_train_X)

            batch_train_ce_loss = F.cross_entropy(logit, batch_train_Y)

            '''
            # equivalent to loss = F.cross_entropy(logit, target)
            log_softmax = F.log_softmax(logit)
            total_batch_loss = F.nll_loss(log_softmax, batch_train_Y, size_average=False)
            cross_entropy = total_batch_loss / data_handler.batch_size
            softmax = t.exp(log_softmax)
            '''

            #------------
            # Loss from generated data

            if use_cuda:
                batch_generated_X = batch_generated_X.cuda()
                batch_generated_Y = batch_generated_Y.cuda()

            batch_generated_X = self.embedding.word_embed(batch_generated_X)
            logit_generated = self.sentiment_discriminator(batch_generated_X)

            log_softmax_generated = F.log_softmax(logit_generated)
            softmax_generated = t.exp(log_softmax_generated)
            generated_loss_ce = F.nll_loss(log_softmax_generated, batch_generated_Y)

            # Calculate emperical shannon entropy from the probs when we use generated data as the input
            # print log_softmax_generated.size()
            # print (t.sum(log_softmax_generated, 1)).size()
            # print (t.mean(t.sum(log_softmax_generated))).size()
            # sys.exit()
            emperical_shannon_entropy = t.neg(t.mean(t.sum(softmax_generated*log_softmax_generated, 1)))
            
            generated_loss = generated_loss_ce + self.config.beta * emperical_shannon_entropy
            
            #--------
            # Total Loss

            total_loss = batch_train_ce_loss + self.config.lambda_u*generated_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            '''
            # TODO: Make entropy calculation generic for all generated_c.
            emperical_shannon_entropy = t.neg(t.mean(t.sum(softmax * log_softmax, 1)))
            
            optimizer.zero_grad()
            loss = cross_entropy + emperical_shannon_entropy
            loss.backward()
            optimizer.step()
            '''
            
            # return the total_batch_loss
            # can be used to calculate the total epoch loss
            return total_loss, batch_train_ce_loss, generated_loss, generated_loss_ce, emperical_shannon_entropy

        return train


    def prep_disc_validation(self):

        """
        Call prep functions before retrieving the step (train/valid) functions
        Otherwise optimizer would not update the weights
        """
        
        # Valid Mode for all modules
        self.discriminator_mode(train_mode=False)
        self.encoder_mode(train_mode=False)
        self.generator_mode(train_mode=False)

        # Already being set in train-encoder-generator 
        self.sentiment_discriminator.eval()
        self.encoder.eval()
        self.generator.eval()

    def prep_disc_training(self):

        # Valid Mode for all modules
        self.discriminator_mode(train_mode=True)
        self.encoder_mode(train_mode=False)
        self.generator_mode(train_mode=False)

        # Already being set in train-encoder-generator 
        self.sentiment_discriminator.train()
        self.encoder.eval()
        self.generator.eval()

    def prep_enc_gen_validation(self):

        self.encoder_mode(train_mode=False)
        self.generator_mode(train_mode=False)
        self.discriminator_mode(train_mode=False)

        self.encoder.eval()
        self.generator.eval()
        self.sentiment_discriminator.eval()

    def prep_enc_gen_training(self):

        # Train Mode off for discriminator
        self.discriminator_mode(train_mode=False)
        self.sentiment_discriminator.eval()
        
        # Train Mode on for encoder and generator
        self.encoder_mode(train_mode=True)
        self.generator_mode(train_mode=True)
        self.encoder.train()
        self.generator.train()
        
    def valid_encoder_generator(self, data_handler):
        
        def valid(start_index, batch_size, use_cuda, dropout, c_target, global_step):

            c_target = c_target[start_index:start_index+batch_size]

            input = data_handler.gen_batch_loader.next_batch(batch_size, 'valid', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, gen_word_input, _ , target] = input
            
            #--------------
                
            [batch_size, _] = encoder_word_input.size()
            encoder_input = self.embedding(encoder_word_input, encoder_character_input)

            z_out, std, logvar, mu = self.encode_for_z(encoder_input, batch_size, use_cuda)

            # Detach the z from the exising graph
            z = Variable(z_out.data, requires_grad=True)
            
            c = Variable(c_target, requires_grad=False)
            if use_cuda:
                c = c.cuda()

            kld_coeff = kld_coef(global_step, extended=True)
            kld = (-0.5 * t.sum(logvar - t.pow(mu,2) - t.exp(logvar) + 1, 1)).mean().squeeze()

            c = c.view(c.size(0), 1)

            input_code = t.cat((z,c), 1)
            generator_input = self.embedding.word_embed(gen_word_input)
            logits, _ = self.generator(generator_input, input_code, dropout, None)
            
            # --------------
            
            logits = logits.view(-1, self.config.word_vocab_size)
            target = target.view(-1)
            
            # cross_entropy = F.cross_entropy(logits, target)
            log_softmax = F.log_softmax(logits)
            total_batch_loss = F.nll_loss(log_softmax, target, size_average=False)
            cross_entropy = 80 * total_batch_loss / batch_size

            # -------------
            
            # use this for z and z_reconstruction loss
            softmax_output = t.exp(log_softmax)
            
            # both are vairables here
            out_embedding = t.mm(softmax_output , self.embedding.word_embed.weight)
            out_embedding = out_embedding.view(batch_size, -1, out_embedding.size(1))[:,:-1,:]

            # use this out_embedding for forward pass to get z loss
            # use this out_mebedding for forward pass to get cross_entropy for c

            # Generator loss using discriminator
            c_reconstructed = self.sentiment_discriminator(out_embedding)
            
            # c_target = c_target.long()
            c = c.long()
            c_loss = F.cross_entropy(c_reconstructed, c.view(-1))

            # Generator loss using encoder
            encoder_character_embedding = self.embedding.get_character_embed(encoder_character_input)
            encoder_reconstructed_input = t.cat([out_embedding, encoder_character_embedding], dim=2)

            # Should i use mu and logvar l2 here? 
            z_recon, std_recon, _, mu_recon = self.encode_for_z(encoder_reconstructed_input, batch_size, use_cuda)            
            
            mahalabonis_distance = self.compute_mahalabonis(mu, mu_recon, std, std_recon)
            z_loss = t.sum(mahalabonis_distance)/batch_size
            
            total_reconstruction_loss = (self.config.lambda_z*z_loss) + (self.config.lambda_c*c_loss)
            
            return cross_entropy, kld, total_reconstruction_loss, z_loss, c_loss, kld_coeff

        return valid

    
    def train_encoder_generator(self, data_handler):

        # Two optimizers
        encoder_optimizer = Adam(self.encoder_params(), self.config.learning_rate)
        generator_optimizer = Adam(self.generator_params(), self.config.learning_rate)

        def train(start_index, batch_size, use_cuda, dropout, c_target, global_step):
            
            encoder_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            
            c_target = c_target[start_index:start_index+batch_size]
            
            input = data_handler.gen_batch_loader.next_batch(batch_size, 'train', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, gen_word_input, _ , target] = input
            
            #--------------

            """
            print 'Begin'
            enc_params = self.encoder.named_parameters()
            for name, p in enc_params:
                print name, p.grad
                break
            """
            
            [batch_size, _] = encoder_word_input.size()
            encoder_input = self.embedding(encoder_word_input, encoder_character_input)

            z_out, std_out, logvar_out, mu_out = self.encode_for_z(encoder_input, batch_size, use_cuda)

            # Detach the z from the existing graph
            z = Variable(z_out.data, requires_grad=True)
            
            c = Variable(c_target, requires_grad=False)
            if use_cuda:
                c = c.cuda()

            kld_coeff = kld_coef(global_step, extended=True)
            kld = (-0.5 * t.sum(logvar_out - t.pow(mu_out,2) - t.exp(logvar_out) + 1, 1)).mean().squeeze()

            c = c.view(c.size(0), 1)

            input_code = t.cat((z,c), 1)
            generator_input = self.embedding.word_embed(gen_word_input)
            logits, _ = self.generator(generator_input, input_code, dropout, None)
            
            # --------------
            
            logits = logits.view(-1, self.config.word_vocab_size)
            target = target.view(-1)
            
            # cross_entropy = F.cross_entropy(logits, target)
            log_softmax = F.log_softmax(logits)
            total_batch_loss = F.nll_loss(log_softmax, target, size_average=False)
            cross_entropy = 80 * total_batch_loss / batch_size
            
            # vae_loss = cross_entropy +  kld # kld_coef = 1
            
            # vae_loss.backward(retain_variables=True)

            cross_entropy.backward(retain_variables=True)

            """
            print 'CrossEntropy Backward called'
            enc_params = self.encoder.named_parameters()
            for name, p in enc_params:
                print name, p.grad
                break
                        """
            
            encoder_grad_z_out = z.grad.data
            # we can backprop encoder_grad_z_out from z_out to get encoder.grad
            
            #-------------VAE Loss Backpropagated-----------
                        
            # use this for z and z_reconstruction loss
            softmax_output = t.exp(log_softmax)
            
            # both are vairables here
            out_embedding = t.mm(softmax_output , self.embedding.word_embed.weight)
            out_embedding = out_embedding.view(batch_size, -1, out_embedding.size(1))[:,:-1,:]

            # use this out_embedding for forward pass to get z loss
            # use this out_mebedding for forward pass to get cross_entropy for c

            # Generator loss using discriminator
            c_reconstructed = self.sentiment_discriminator(out_embedding)
            
            # c_target = c_target.long()
            c = c.long()
            c_loss = F.cross_entropy(c_reconstructed, c.view(-1))

            # Generator loss using encoder
            encoder_character_embedding = self.embedding.get_character_embed(encoder_character_input)
            encoder_reconstructed_input = t.cat([out_embedding, encoder_character_embedding], dim=2)
            
            z_recon, std_recon, _, mu_recon = self.encode_for_z(encoder_reconstructed_input, batch_size, use_cuda)

            # Create new mu, and std fro mu_out and std_out
            # To avoid calculating gradient wrt mu_out and std_out when calculating mahalabonis_distance
            mu = Variable(mu_out.data, requires_grad=False)
            std = Variable(std_out.data, requires_grad=False)
            mahalabonis_distance = self.compute_mahalabonis(mu, mu_recon, std, std_recon)

            """
            l2_loss= z-z_recon
            l2_loss = l2_loss*l2_loss
            l2_loss = t.sum(l2_loss, dim=1)
            
            z_loss = t.sum(t.sqrt(l2_loss))/batch_size
            """

            z_loss = t.sum(mahalabonis_distance)/batch_size
            total_reconstruction_loss = (self.config.lambda_z*z_loss) + (self.config.lambda_c*c_loss)
            total_reconstruction_loss.backward()
            generator_optimizer.step()

            """
            print 'Encoder Gradient before Zero'
            enc_params = self.encoder.named_parameters()
            for name, p in enc_params:
                print name, p.grad
                break
            """
            
            encoder_optimizer.zero_grad()

            """
            print 'Encoder Gradient After Zero'
            enc_params = self.encoder.named_parameters()
            for name, p in enc_params:
                print name, p.grad
                break
            """
            
            z_out.backward(encoder_grad_z_out, retain_variables=True)
            kld_loss = kld*kld_coeff
            kld_loss.backward()

            """
            print 'Encoder Gradient Final'
            enc_params = self.encoder.named_parameters()
            for name, p in enc_params:
                print name, p.grad
                break

            sys.exit()
            """
            
            encoder_optimizer.step()
            
            return cross_entropy, kld, total_reconstruction_loss, z_loss, c_loss, kld_coeff 

        return train

    
    def initial_rvae_trainer(self, data_handler):

        self.encoder.train()
        self.generator.train()
        
        optimizer = Adam(self.learnable_parameters(), self.config.learning_rate)
        def train(i, batch_size, use_cuda, dropout, start_index):

            input = data_handler.gen_batch_loader.next_batch(batch_size, 'train', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input,_, target] = input

            logits, _, kld,_ ,_ ,_ = self.train_rvae(dropout,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input)

            logits = logits.view(-1, self.config.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            loss = 79 * cross_entropy + kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, kld, kld_coef(i), loss

        return train

    
    def initial_rvae_valid(self, data_handler):

        # Should remove the dropout scaling
        # TODO: Do this inside the valid function
        self.encoder.eval()
        self.generator.eval()
        
        def valid(i, batch_size, use_cuda, dropout, start_index):

            input = data_handler.gen_batch_loader.next_batch(batch_size, 'valid', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input,_, target] = input

            logits, _, kld,_ ,_ ,_ = self.train_rvae(0,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input)

            logits = logits.view(-1, self.config.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            loss = 79 * cross_entropy + kld_coef(i) * kld
            return cross_entropy, kld, kld_coef(i), loss

        return valid

    def encoder_params(self):

        enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
        enc_params.extend([p for p in self.embedding.parameters() if p.requires_grad])
        enc_params.extend([p for p in self.e2mu.parameters() if p.requires_grad])
        enc_params.extend([p for p in self.e2logvar.parameters() if p.requires_grad])

        return enc_params
    
    def generator_params(self):
        return [p for p in self.generator.parameters() if p.requires_grad]

    def encoder_mode(self, train_mode=False):

        """
        Store the encoder gradient only if train_mode=True
        Uses less memory when training the generator 
        """
        
        print 'Encoder Train Mode: %d' % train_mode

        for i, param in enumerate(self.encoder.parameters()):

            if train_mode:
                if self.encoder_param_autograd[i]:
                    param.requires_grad = True
            else:
                param.requires_grad = False

        for i, param in enumerate(self.embedding.parameters()):

            if train_mode:
                if self.embedding_param_autograd[i]:
                    param.requires_grad = True
            else:
                param.requires_grad = False

        for i, param in enumerate(self.e2mu.parameters()):

            if train_mode:
                param.requires_grad = True
            else:
                param.requires_grad = False

        
        for i, param in enumerate(self.e2logvar.parameters()):

            if train_mode:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        
    def discriminator_mode(self, train_mode=True):

        print 'Discriminator Train: %d' % train_mode
        for i, param in enumerate(self.sentiment_discriminator.parameters()):
            
            if train_mode:
                if self.sentiment_discriminator_autograd[i]:
                    param.requires_grad = True
            else:
                param.requires_grad = False

    def generator_mode(self, train_mode=True):

        print 'Generator Train Mode: %d' % train_mode

        for i, param in enumerate(self.generator.parameters()):
            
            if train_mode:
                if self.generator_param_autograd[i]:
                    param.requires_grad = True
            else:
                param.requires_grad = False

    def sample_from_generator (self, batch_loader, seq_len, seed,
                               use_cuda, beam_size = 10, n_best = 1, samples=5,
                               learning = False):

        """
        Samples sentences based on the current state of the generator.

        Parameters:
        * batch_loader: gen_batch_loader from data_handler.
        * seq_len: max sequence length
        * seed: (z+c) latent code, where c is the structured code and z is the random sample
        * use_cuda: cuda flag
        * beam_size: number of candidate beams to consider for joint probability
        * n_best: n best beams per input to return. Should be 1 while training.
        * samples: number of samples to return. Getting more sample increases memory usage by the order of beam_size
        * learning: is generator learning. 

        Return:
        * all_sent_probs: soft distributon, if learning is true 
        * all_sentences, all_scores: (sentence in words, join probability scores), if learning is false
        
        """

        # self.generator in eval?

        self.generator.eval()
        
        seed = Variable(seed)
        if use_cuda:
            seed = seed.cuda()

        # State see the shape
        dec_states = None
        drop_prob = 0.0
        batch_size = samples # Make this samples
        
        beam = [Beam(beam_size, batch_loader, cuda=True) for k in range(batch_size)]
        
        batch_idx = list(range(batch_size))
        remaining_sents =  batch_size
        
        for i in range(seq_len):
            
            input = t.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)
            # input becomes (1, beam_size * batch_size)

            trg_emb = self.embedding.word_embed(Variable(input).transpose(1, 0))
            # trg_emb.size() => (beam_size*batch_size, 1, embedding_size)
                        
            trg_h, dec_states = self.generator.only_decoder_beam(trg_emb, seed, drop_prob, dec_states)
            # trg_h.size() => (beam_size*batch_size, 1, gen_rnn_size)
            # dec_states => tuple of hidden states and cell state
            
            dec_out = trg_h.squeeze(1)
            # dec_out.size() => (beam_size*batch_size, generator_rnn_size)
            
            out = F.log_softmax(self.generator.fc(dec_out)).unsqueeze(0)
            # out.size() => (1, beam_size*batch_size, vocab_size)
                        
            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()
            # word_lk.size() => (remaining_sents, beam_size, vocab_size)

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                # beam state advance
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]
                
                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        dec_state.size(0), beam_size, remaining_sents, dec_state.size(2)
                    )[:,:,idx,:]

                    
                    # sent_states.size() => (layers, beam_size, gen_rnn_size)

                    
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )
                
            if not active:
                break
            
            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = t.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # t.size() => (beam*remaining_sentences, decoder_rnn_size)
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.config.decoder_rnn_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
                        
            # print 'Remaining Sents: %d ' % remaining_sents
            dec_out = update_active(dec_out)
            remaining_sents = len(active)
            

         # (4) package everything up
                
        allHyp, allScores = [], []
        allHyp_probs = []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            hyp_probs = beam[b].get_hyp_probs() 
            allHyp += [hyps]
            allHyp_probs += [hyp_probs]
            
        word_hyp = []
        result = []
                
        all_sent_codes = np.asarray(allHyp)
        all_sent_probs = np.asarray(allHyp_probs)

        #print all_sent_probs.shape

        #print 'prob sen 1 size: %d' % len(all_sent_probs[0])
        #print len(all_sent_probs[0][2])
        #s = np.argsort(np.array(all_sent_probs[0][2].tolist()))[-10:]
        
        #print list(map(batch_loader.decode_word, s))
        #print all_sent_codes[0][0]
        #all_sent_codes = np.transpose(allHyp, (0,2,1))
        #print all_sent_codes

        if learning:
            return all_sent_probs
        else :
            """
            all_sentences = []
            for batch in all_sent_codes:
                sentences = []

                for i_best in batch:
                    sentence = ""
                    for word_code in i_best:
                        word = batch_loader.decode_word(word_code)
                        if word == batch_loader.end_token:
                            break
                        sentence += ' ' + word
                    sentences.append(sentence)

                all_sentences.append(sentences)
            """
            
            return all_sent_codes, allScores 

    def sample(self, data_handler, config, use_cuda=True, print_sentences = True):

        """
        This function uses the current state of the generator to sample.
        
        Parameters:
        * data_handler: Data handler to create batch
        * config: model configuration object
        * use_cuda: cuda flag

        Returns:
        * generated_samples: softmax outputs of the generator
        * seed_c: the corresponding structured codeword
        """

        self.generator.eval()
        
        samp = 5
        seed_z = t.randn([samp, config.latent_variable_size])
        seed_z = seed_z.unsqueeze(1)
        seed_z = seed_z.repeat(1, 2, 1)
        seed_z = seed_z.view(-1, config.latent_variable_size)
        
        #init_prob = t.ones(samp, 1)*0.5
        #seed_c = t.bernoulli(init_prob)
        seed_c = t.ones(2*samp, 1)
        for i in range(samp):
            seed_c[2*i+1] = 0
        
        seed = t.cat((seed_z, seed_c), 1)

        print seed.size()
        
        sentences, result_score = self.sample_from_generator(data_handler.gen_batch_loader, config.max_seq_len, seed, use_cuda, n_best = 1, samples=2*samp)

        if print_sentences:
            print len(sentences)

            seed_c = seed_c.numpy()
            i = 0
            for s in (sentences):
                sen = ""
                for word in s:
                    sen += data_handler.gen_batch_loader.decode_word(word[0]) + ' '
                print 'c={%d} : %s' % (seed_c[i], sen)
                i += 1

            """
            for batch in sentences:
                for i_best in batch:
                    sentence = ""
                    for word_code in i_best:
                        word = batch_loader.decode_word(word_code)
                        if word == batch_loader.end_token:
                            break
                        sentence += ' ' + word
                    print sentence
            """
            
        else:

            all_sentences = []
            for batch in generated_samples:
                sentence = []
                for word_n in batch:
                    sentence.append(word_n[0])
                all_sentences.append(sentence)

            return all_sentences, seed_c

    def sample_generator_for_learning(self, data_handler, config, use_cuda = True, sample=10):

        """
        This function uses the current state of the generator to sample.
        
        Parameters:
        * data_handler: Data handler to create batch
        * config: model configuration object
        * use_cuda: cuda flag
        * sample: number of samples to generate

        Returns:
        * all_gen_samples: softmax outputs of the generator
        * seed_c: the corresponding structured codeword
        """

        self.generator.eval()
        
        seed_z = t.randn([sample, config.latent_variable_size])
        init_prob = t.ones(sample, 1) * 0.5
        seed_c = t.bernoulli(init_prob)

        seed = t.cat((seed_z, seed_c), 1)

        generated_samples, _ = self.sample_from_generator(data_handler.gen_batch_loader, config.max_seq_len, seed, use_cuda, n_best=1,  samples=sample, learning=False)
        
        all_gen_samples = []
        for batch in generated_samples:
            sentence = []
            for word_n in batch:
                sentence.append(word_n[0])
            all_gen_samples.append(sentence)

        """
        generated_samples = data_handler.create_generator_batch(all_sentences, use_cuda)

        if use_cuda:
            seed_c = seed_c.cuda()
            generated_samples = generated_samples.cuda()
        """
        
        return all_gen_samples, seed_c
        
    def sentiment_discriminator_parameters(self):
        return [p for p in self.sentiment_discriminator.parameters() if p.requires_grad]
    
    def learnable_parameters(self):
        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]    
