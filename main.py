__author__ = "Nikhil Mehta"
__copyright__ = "--"
#---------------------------

import argparse
import os
import sys
import re

import numpy as np
import torch as t

from utils.data_handler import DataHandler
from utils.config import Config

from model.cgn import Controlled_Generation_Sentence

import torchvision
from tensorboardX import SummaryWriter

ROOT_DIR = '/data1/nikhil/sentence-corpus/'

def train_encoder_decoder(cgn_model, config, data_handler, num_epochs, use_cuda, dropout, summary_writer, total_steps):
    """
    Trains encoder-generator while keeping the discriminator fixed using the following loss functions:
    - encoder_loss = VAE Loss = KLD + Cross Entropy Loss
    - generator_loss = VAE Loss + disc_coeff*disc_cross_entropy + encoder_coeff*Encoder L2 Loss  

    Parameters:

    * cgn_model: controllable generation model instance
    * config: config instance created in main.py 
    * data_handler: Manages all data/batch creation 
    * num_epochs: Train enc-gen for num_epochs
    * use_cuda: cuda flag
    * dropout: dropout probability
    * summary_writer: tensorboard summary writer
    * total_steps: steps for which enc-gen have already been trained
    """

    # First fix the discriminator. Switch of the gradients by setting autograd to False
    # Get codewords for the generator dataset. This will take time if VAE train-set is large.
    # Do VAE training with the calculated codeword. Store the z generated.
    # Pass the generated sentence softmax output to discriminator and backprop the gradient to update the generator weights
    # Pass the generated sentence softmax output back to encoder and use L2 loss for z 

    cgn_model.discriminator_mode(train_mode=False)

    #-------------------Get codewords from the current state of discriminator-------- 

    pass_gradient_to_generator = False
    batch_size = 500
    discriminator_forward = cgn_model.discriminator_forward_function (data_handler, use_cuda, pass_gradient_to_generator, batch_size)
        
    num_line = (data_handler.gen_batch_loader.num_lines[0])
    num_line = num_line - num_line % batch_size
    num_batches = num_line / batch_size

    c = []
    for batch_index in range(num_batches):
        c_batch = discriminator_forward(batch_index)
        c.append(c_batch)
        if batch_index % 50 == 0:
            print 'Batch: %d' % batch_index

    c = t.stack(c, dim=0)
    c = c.view(-1)
        
    #----------------------Now Train Encoder-Generator--------------------------------

    # Use a smaller batch_size
    num_line = data_handler.gen_batch_loader.num_lines[0]
    num_line = num_line - num_line % data_handler.batch_size
    num_batches = num_line / data_handler.batch_size

    train_enc_gen = cgn_model.train_encoder_generator(data_handler)
    
    pass_gradient_to_generator = True

    for epoch in range(0, num_epochs):
        for batch_index in range(num_batches):

            iteration = epoch*num_batches + batch_index
            vae_loss, cross_entropy, kld, gen_loss = train_enc_gen(batch_index,
                                                     data_handler.batch_size,
                                                     use_cuda, dropout,
                                                     c_target=c)

            vae_loss_val = vae_loss.data.cpu()[0]
            ce_loss_val = cross_entropy.data.cpu()[0]
            kld_val = kld.data.cpu()[0]
            gen_loss_val = gen_loss.data.cpu()[0]
                        
            if total_steps % 50 == 0:
                
                print('\n')
                print ('------------------------')
                print ('Encoder-Generator Data')
                print ('Epoch: %d Batch_Index: %d/%d' % (epoch+1, batch_index, num_batches))
                print ('Enc-Gen Training Step: %d' % total_steps)
                print ('Encoder Loss: %f' % vae_loss_val)
                print ('Generator Loss: %f' % (vae_loss_val+gen_loss_val))
                print ('Cross Entropy: %f' % ce_loss_val)
                print ('KLD: %f' % kld_val)
                
                summary_writer.add_scalar('train_enc_gen/encoder_loss', vae_loss_val, total_steps)
                summary_writer.add_scalar('train_enc_gen/generator_loss', (vae_loss_val+gen_loss_val), total_steps)
                summary_writer.add_scalar('train_enc_gen/cross_entropy', ce_loss_val, total_steps)
                summary_writer.add_scalar('train_enc_gen/kld', kld_val, total_steps)
                
            total_steps += 1
            
    return total_steps

def train_sentiment_discriminator(cgn_model, config, data_handler, num_epochs, use_cuda, summary_writer, total_training_steps):

    """
    Trains the discriminator using both the labeled data and unlabedled data (from the generator). 

    Parameters:
    * cgn_model: A cgn_model with pretrained generator and encoder weights. (See main() flags).
    * data_handler: data_handler instance that manages all the data and batch preparation
    * num_epochs: The number of epochs for which the discriminator is trained in this sleep phase
    * use_cuda: cuda flag

    """

    print 'Training Sentiment Discriminator. Total Gen-Disc Steps: %d' % total_training_steps     
        
    num_train_batches = data_handler.num_train_sentiment_batches

    number_of_gen_samples = data_handler.batch_size # Number of samples to generate
    gen_samples_batch_size = 10  # Generate batch_size samples in 1 call. Increasing batch_size here will increase memory consumption proportional to O(beam_size)

    sentiment_disc_train_step = cgn_model.discriminator_sentiment_trainer(data_handler, use_cuda)

    for epoch in range(0, num_epochs):

        epoch_loss = 0

        for batch_index in range(num_train_batches):

            generated_samples, generated_c = [], []
            for ith_sample in range(number_of_gen_samples/gen_samples_batch_size):
                gen_sample, gen_c = cgn_model.sample_generator_for_learning(data_handler, config, use_cuda, sample=gen_samples_batch_size)
                generated_samples.extend(gen_sample)
                generated_c.extend(gen_c)

            generated_c = t.stack(generated_c, dim=0)
            batch_loss, batch_cross_entropy, batch_emperical_shannon_entropy = sentiment_disc_train_step(generated_samples, generated_c, batch_index)

            loss_val = batch_loss.data.cpu()[0]
            ce_loss_val = batch_cross_entropy.data.cpu()[0]
            ese_loss_val = batch_emperical_shannon_entropy.data.cpu()[0]
            
            epoch_loss += loss_val
                        
            if total_training_steps % 50 == 0:
                
                print('\n')
                print ('------------------------')
                print ('Discriminator Training')
                print ('Epoch: %d Batch: %d/%d' % (epoch+1, batch_index, num_train_batches))
                print ('Total Training Steps: %d' % total_training_steps)
                print ('Batch loss: %f' % loss_val)
                print ('Batch cross Entropy: %f' % ce_loss_val)
                print ('Batch emperical shannon entropy: %f' % ese_loss_val)  

                summary_writer.add_scalar('train_discriminator/total_loss', loss_val, total_training_steps)
                summary_writer.add_scalar('train_discriminator/cross_entropy', ce_loss_val, total_training_steps)
                summary_writer.add_scalar('train_discriminator/cross_entropy', ce_loss_val, total_training_steps)
                                
            total_training_steps += 1

        epoch += 1

    # step helps keep track of generator-discriminator alternating iterations
    return total_training_steps

def main():

    parser = argparse.ArgumentParser(description='Controlled_Generation_Sentence')

    parser.add_argument('--rvae-initial-iterations', type=int, default=120000, help="initial rvae training steps")
    parser.add_argument('--discriminator-epochs', type=int, default=1, help="num epochs for which disc is to be trained while alternating")
    parser.add_argument('--generator-epochs', type=int, default=1, help="num epochs for which gener is to be trained while alternating")
    parser.add_argument('--total-alternating-iterations', type=int, default=1000, help="number of alternating iterations")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--sample-generator', type=bool, default=False)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--generator-train-file', type=str, default=os.path.join(ROOT_DIR, 'raw_train_generator.pkl'))
    parser.add_argument('--embedding-path', type=str, default=os.path.join(ROOT_DIR, 'word_embeddings.npy'))
    parser.add_argument('--preload-initial-rvae', type=str, default=None)
    parser.add_argument('--save-model-dir', type=str, default=os.path.join(ROOT_DIR, 'snapshot/'))
    parser.add_argument('--words-vocab-path', type=str, default=os.path.join(ROOT_DIR, 'words_vocab.pkl'))
    parser.add_argument('--chars-vocab-path', type=str, default=os.path.join(ROOT_DIR, 'characters_vocab.pkl'))
    parser.add_argument('--sentiment-discriminator-train-file', type=str, default=os.path.join(ROOT_DIR, 'raw_train_sentiment_discriminator.pkl'))
    parser.add_argument('--train-cgn-model', type=bool, default=False)
    
    args = parser.parse_args()

    if not os.path.exists(args.embedding_path):
        raise FileNotFoundError("Word Embedding FileNotFoundError")

    if not os.path.exists(args.words_vocab_path):
        raise FileNotFoundError("Words Vocabulary FileNotFoundError")

    if not os.path.exists(args.chars_vocab_path):
        raise FileNotFoundError("Characters Vocabulary FileNotFoundError")

    if not os.path.exists(args.generator_train_file):
        raise FileNotFoundError("Generator Training FileNotFoundError")

    if not os.path.exists(args.sentiment_discriminator_train_file):
        raise FileNotFoundError("Sentiment Training FileNotFoundError")

    if not os.path.exists(args.save_model_dir):
        try:
            os.makedirs(args.save_model_dir)
        except OSError as e:
            raise OSError('Directory can\'t be created')
    
    vocab_files = [args.words_vocab_path, args.chars_vocab_path]

    train_initial_rvae = args.preload_initial_rvae == None
    data_handler = DataHandler(vocab_files, args.generator_train_file, args.batch_size)    
    
    config = Config(data_handler.gen_batch_loader.max_word_len, data_handler.gen_batch_loader.max_seq_len, data_handler.gen_batch_loader.words_vocab_size, data_handler.gen_batch_loader.chars_vocab_size, args.learning_rate)
    
    cgn_model = Controlled_Generation_Sentence(config, args.embedding_path)

    if args.use_cuda:
        cgn_model = cgn_model.cuda()

        
    if train_initial_rvae:

        summary_dir = ROOT_DIR+'snapshot/run_initial/'
        summary_writer = SummaryWriter(summary_dir)

        start_iteration = 0

        
        initial_train_step = cgn_model.initial_rvae_trainer(data_handler)
        # initial_validate_step ?

        start_index = 0
        ce_result = 0
        kld_result = 0

        num_line = (data_handler.gen_batch_loader.num_lines[0])
        num_line = num_line - num_line % args.batch_size
        print 'Begin Step 1. Initial VAE Training'

        for iteration in range(start_iteration, args.rvae_initial_iterations):

            start_index = (start_index+args.batch_size)%num_line
            cross_entropy, kld, coef, total_loss = initial_train_step(iteration, args.batch_size, args.use_cuda, args.dropout, start_index)

            if iteration % 50 == 0:
                ce_loss_val = cross_entropy.data.cpu()[0]
                kld_loss_val = kld.data.cpu()[0]
                total_loss_val = total_loss.data.cpu()[0]

                print('\n')
                print ('------------------------')
                print ('Iteration: %d'%iteration)
                print ('Cross entropy: %f'%ce_loss_val)
                print ('KLD: %f'%kld_loss_val)
                print ('KLD Coef: %f' % coef)
                print ('Total Loss: %f'%(total_loss_val))
                
                summary_writer.add_scalar('train_initial_rvae/kld_coef', coef, iteration)
                summary_writer.add_scalar('train_initial_rvae/kld', kld_loss_val, iteration)
                summary_writer.add_scalar('train_initial_rvae/cross_entropy', ce_loss_val, iteration)                
                summary_writer.add_scalar('train_initial_rvae/total_loss', (total_loss_val), iteration)
                
        t.save(cgn_model.state_dict(), os.path.join(args.save_model_dir, 'initial_rvae'))

    elif (args.sample_generator) :

        # load a pretrained model if needed
        cgn_model.load_state_dict(t.load(args.preload_initial_rvae))
        print 'Initial Model Loaded'
        cgn_model.sample(data_handler, config, args.use_cuda)

    elif (args.train_cgn_model):

        summary_dir = ROOT_DIR+'snapshot/train_cgn_logs/'
        summary_writer = SummaryWriter(summary_dir)
        
        # Load pretrained
        cgn_model.load_state_dict(t.load(args.preload_initial_rvae))

        # Load discriminator labelled data
        data_handler.load_discriminator(args.sentiment_discriminator_train_file)

        total_discriminator_steps = 0
        total_generator_steps = 0

        for i in range(args.total_alternating_iterations):
            
            total_discriminator_steps = train_sentiment_discriminator(cgn_model, config, data_handler, args.discriminator_epochs,
                                                                      args.use_cuda, summary_writer, total_discriminator_steps)
            
            total_generator_steps = train_encoder_decoder(cgn_model, config, data_handler, args.generator_epochs,
                                                          args.use_cuda, args.dropout, summary_writer, total_generator_steps)

        t.save(cgn_model.state_dict(), os.path.join(args.save_model_dir, 'cgn_model'))

        summary_writer.export_scalars_to_json(summary_dir+"all_scalars.json")
    else:

        summary_writer = SummaryWriter(ROOT_DIR+'snapshot/run_random_logs/')

        
        # cgn_model.load_state_dict(t.load(args.preload_initial_rvae))
        # data_handler.load_discriminator(args.sentiment_discriminator_train_file)
        # train_sentiment_discriminator(cgn_model, config, data_handler, args.discriminator_epochs, args.use_cuda)
        # train_encoder_decoder (cgn_model, config, data_handler, args.generator_epochs, args.use_cuda, args.dropout)

    
    summary_writer.close()

    
if __name__ == '__main__':
    main()
