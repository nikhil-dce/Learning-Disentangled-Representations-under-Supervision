__author__ = "Nikhil Mehta"
__copyright__ = "--"

import numpy as np
import torch as t
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

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

    def train_rvae (self, drop_prob, encoder_word_input=None, encoder_char_input=None, generator_word_input=None):

        use_cuda = self.embedding.word_embed.weight.is_cuda
        
        [batch_size, _] = encoder_word_input.size()
        encoder_input = self.embedding(encoder_word_input, encoder_char_input)

        context, h_0, c_0 = self.encoder(encoder_input, None)
        State = (h_0, c_0)

        mu = self.e2mu(context)
        logvar = self.e2logvar(context)

        std = t.exp(0.5*logvar)

        z = Variable(t.randn([batch_size, self.config.latent_variable_size]))

        if use_cuda:
            z = z.cuda()
            
        z = z * std + mu
        
        init_prob = t.ones(batch_size, 1)*0.5
        c = Variable(t.bernoulli(init_prob), requires_grad=False)

        if use_cuda:
            c = c.cuda()
        
        input_code = t.cat((z,c), 1)
                    
        kld = (-0.5 * t.sum(logvar - t.pow(mu,2) - t.exp(logvar) + 1, 1)).mean().squeeze()

        generator_input = self.embedding.word_embed(generator_word_input)
        out, final_state = self.generator(generator_input, input_code, drop_prob, None)

        return out, final_state, kld, mu, std, z.data

    def discriminator_forward_function (self, data_handler, use_cuda, pass_gradient_to_generator, batch_size):

        # batch_size here can be increased for faster forward propagation

        self.sentiment_discriminator.set_pass_gradient_to_generator(pass_gradient_to_generator)
        self.sentiment_discriminator.eval()
        
        def discriminator_forward(batch_index):
            
            input = data_handler.gen_batch_loader.next_batch(batch_size, 'train', batch_index)
            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            encoder_word_input = t.from_numpy(encoder_word_input)
            sentence_hot_input = data_handler.feature_from_indices(encoder_word_input)
            sentence_hot_input = Variable(sentence_hot_input)
            
            if use_cuda:
                sentence_hot_input = sentence_hot_input.cuda()
                        
            logit = self.sentiment_discriminator(sentence_hot_input)

            softmax = F.softmax(logit)
            
            # bernoulli here with softmax[:,1]
            c = t.bernoulli(softmax[:, 1])
                        
            return c.data
            
        return discriminator_forward
        
    def discriminator_sentiment_trainer (self, data_handler, use_cuda):

        # Set train mode. Do not pass gradient to the generator
        self.sentiment_discriminator.train()

        # print self.sentiment_discriminator.state_dict()
        # for name, param in self.named_parameters():
            # print name + ' ' + str(param.requires_grad)
        
        #self.encoder_params(auto_grad=False)
        #self.generator_params(auto_grad=False)
        
        # get the relevant model parameters
        optimizer = t.optim.Adam(self.sentiment_discriminator_parameters(), self.config.learning_rate)
               
        def train(i, batch_index):

            # This should give samples from both the generator and discriminator dataset
            # These should be torch tensors as one-hot vectors (discriminator dataset) or softmax outputs (Generated sentences)
            # expected dimenstion is (batch_size, seq_len, vocab_size)
            batch_train_X, batch_train_Y = data_handler.get_sentiment_train_batch(batch_index)

            batch_size = batch_train_X.size(0)
            batch_train_X = batch_train_X.view(-1, data_handler.gen_batch_loader.words_vocab_size)
            
            # Torch tensors
            if use_cuda:
                batch_train_X = batch_train_X.cuda()
                batch_train_Y = batch_train_Y.cuda()
            
            batch_train_X = t.mm(batch_train_X, self.embedding.word_embed.weight.data)
            batch_train_X = batch_train_X.view(batch_size, -1, self.embedding.word_embed.weight.size(1))
                        
            batch_train_Y = Variable(batch_train_Y)
            batch_train_X = Variable(batch_train_X)
            
            # Check if batch_train_X and batch_train_Y are autograd variables
            # make them cuda if needed
            # check if the autograd needs to True. Should be true when training generator
            
            optimizer.zero_grad()
            logit = self.sentiment_discriminator(batch_train_X)

            # equivalent to loss = F.cross_entropy(logit, target)
            log_softmax = F.log_softmax(logit)
            total_batch_loss = F.nll_loss(log_softmax, batch_train_Y, size_average=False)
            loss = total_batch_loss / data_handler.batch_size

            loss.backward()
            optimizer.step()

            # return the total_batch_loss
            # can be used to calculate the total epoch loss
            return total_batch_loss

        return train

    def train_encoder_generator(self, data_handler):

        self.encoder_mode(train_mode=True)
        self.generator_mode(train_mode=True)
        
        # Two optimizers
        encoder_optimizer = Adam(self.encoder_params(), self.config.learning_rate)
        generator_optimizer = Adam(self.generator_params(), self.config.learning_rate)

        def train(step, batch_index, batch_size, use_cuda, dropout):

            # Turn the encoder training back on
            self.encoder_mode(train_mode=True)
            encoder_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            
            input = data_handler.gen_batch_loader.next_batch(batch_size, 'train', batch_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            print target.size()
            print target[0]
            
            # fix the encoder, embedding, e2mu, e2logvar
            logits, _, kld,_ ,_, z = self.train_rvae(dropout,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input)

            print logits.size()
            logits = logits.view(-1, self.config.word_vocab_size)
            target = target.view(-1)
            
            # cross_entropy = F.cross_entropy(logits, target)

            log_softmax = F.log_softmax(logits)
            print 'Log Softmax ' + str(log_softmax.size()) + ' Target: ' + str(target.size())
            
            total_batch_loss = F.nll_loss(log_softmax, target, size_average=False)
            cross_entropy = total_batch_loss / data_handler.batch_size

            vae_loss = 79 * cross_entropy +  kld # kld_coef = 1
            # this 79 coeff is not needed. TODO: Remove it

            vae_loss.backward(retain_variables=True)

            #-------------VAE Loss Backpropagated-----------
            # Both Encoder.grad and Generator.grad should be set wrt vae_loss
            
            # Disable Encoder Training. Encoder.grad is still set.
            self.encoder_mode(train_mode=False)

            # use this for z and z reconstruction loss
            softmax_output = t.exp(log_softmax)
            #softmax_output = softmax_output.view(batch_size, -1, self.config.word_vocab_size)
            print softmax_output.size()
            print type(self.embedding.word_embed.weight), self.embedding.word_embed.weight.size()
            
            out_embedding = t.mm(softmax_output , self.embedding.word_embed.weight.data)
            print out_embedding.size()

            out_embedding = out_embedding.view(batch_size, -1)
            
            #sen1 = softmax_output[0]
            #_, indices = sen1.max(1)

            #print indices
            #print indices.size()
            #print softmax_output.size()

            encoder_optimizer.step()
            generator_optimizer.step()
            
            return cross_entropy, kld, 1

        return train
    
    def initial_rvae_trainer(self, data_handler):
        
        optimizer = Adam(self.learnable_parameters(), self.config.learning_rate)
        def train(i, batch_size, use_cuda, dropout, start_index):

            input = data_handler.gen_batch_loader.next_batch(batch_size, 'train', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

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

                
    def sentiment_discriminator_parameters(self):
        return [p for p in self.sentiment_discriminator.parameters() if p.requires_grad]
    
    def learnable_parameters(self):
        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def sample_from_generator (self, batch_loader, seq_len, seed,
                               use_cuda, beam_size = 10, n_best = 1, samples=5,
                               learning = False):
        
        seed = Variable(seed)
        if use_cuda:
            seed = seed.cuda()

        # seed = seed.unsqueeze(1)
        # seed = t.cat([seed] * beam_size, 1)
        # print seed
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
        
            return all_sentences, allScores 

    def sample(self, data_handler, config, use_cuda=True, print_sentences = True):

        samp = 10
        seed_z = t.randn([samp, config.latent_variable_size])
        init_prob = t.ones(samp, 1)*0.5
        seed_c = t.bernoulli(init_prob)
        
        seed = t.cat((seed_z, seed_c), 1)

        print seed.size()
        
        sentences, result_score = self.sample_from_generator(data_handler.gen_batch_loader, config.max_seq_len, seed, use_cuda, n_best = 1, samples=samp)

        if print_sentences:
            print len(sentences)
            for s in sentences:
                sen = ""
                for word in s:
                    sen += word
                print sen

    def sample_generator_for_learning(self, data_handler, config, use_cuda = True):

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
        sample = 10
        seed_z = t.randn([sample, config.latent_variable_size])
        init_prob = t.ones(sample, 1) * 0.5
        seed_c = t.bernoulli(init_prob)

        seed = t.cat((seed_z, seed_c), 1)

        generated_samples = self.sample_from_generator(data_handler.gen_batch_loader, config.max_seq_len, seed, use_cuda, n_best=1,  samples=sample, learning=True)
        generated_samples = data_handler.create_generator_batch(generated_samples, use_cuda)

        if use_cuda:
            seed_c = seed_c.cuda()
            
        return generated_samples, seed_c
        
        
