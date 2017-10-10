__author__ = "Nikhil Mehta"
__copyright__ = "--"

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

"""
TODO
Create separate training params for generator and discriminator and then average them
This should help analyze the training progress 
"""

def train_encoder_decoder_wake_phase(cgn_model, config, data_handler, num_epochs, use_cuda, dropout):
    """
    Trains encoder-generator while keeping the discriminator fixed using the following loss functions:
    - encoder using Loss = VAE Loss = KLD + Cross Entropy Loss
    - generator using Loss = VAE Loss + disc_coeff*Discriminator Cross Entropy Using generated sentence softmax + encoder_coeff*Encoder L2 Loss  
    """

    # First fix the discriminator. Switch of the gradients by setting autograd to False
    # Get codewords for the generator dataset. This will take time if VAE train-set is large.
    # Do VAE training with the calculated codeword. Store the z generated.
    # Pass the generated sentence softmax output to discriminator and backprop the gradient to update the generator weights
    # Pass the generated sentence softmax output back to encoder and use L2 loss for z 

    cgn_model.discriminator_mode(train_mode=False)

    #-------------------Get codewords from the current state of discriminator-------- 

    """
    pass_gradient_to_generator = False
    batch_size = 500
    discriminator_forward = cgn_model.discriminator_forward_function (data_handler, use_cuda, pass_gradient_to_generator, batch_size)
        
    num_line = (data_handler.gen_batch_loader.num_lines[0])
    num_line = num_line - num_line % batch_size
    num_batches = num_line / batch_size

    print 'Num batches: %d' % num_batches 
    c = []
    for batch_index in range(num_batches):
        c_batch = discriminator_forward(batch_index)
        c.append(c_batch)
        if batch_index % 50 == 0:
            print 'Batch: %d' % batch_index

    print len(c)
    """

    #----------------------Now Train Generator--------------------------

    # Use a smaller batch_size
    num_line = data_handler.gen_batch_loader.num_lines[0]
    num_line = num_line - num_line % data_handler.batch_size
    num_batches = num_line / data_handler.batch_size

    train_enc_gen = cgn_model.train_encoder_generator(data_handler)
    
    pass_gradient_to_generator = True

    step = 0
    for epoch in range(0, num_epochs):

        epoch_loss = 0

        for batch_index in range(num_batches):

            iteration = epoch*num_batches + batch_index
            cross_entropy, kld, coef = train_enc_gen(step, batch_index,
                                                     data_handler.batch_size,
                                                     use_cuda, dropout)
                                                     
            sys.exit(0)
            # returns tensor with total_batch_loss
            total_batch_loss = sentiment_disc_train_step(step, batch_index)

            loss_val = total_batch_loss.data[0]
            epoch_loss += loss_val
                        
            if step % 50 == 0:
                
                print('\n')
                print ('------------------------')
                print ('Epoch: %d Batch_Index: %d' % (epoch, batch_index))
                print ('Total Training Steps: %d' % step)
                print ('Batch Loss: %f' % loss_val)

            step += 1

def train_sentiment_discriminator(cgn_model, config, data_handler, num_epochs, use_cuda):

    """
    Trains the discriminator using both the labeled data and unlabedled data (from the generator). This semi-supervised 
    setting involving generator data allows training of both the generator and the discriminator in an wake-sleep 
    fashion. This function is used in the sleep phase.

    Parameters:
    * cgn_model: A cgn_model with pretrained generator and encoder weights. (See main() flags).
    * data_handler: data_handler instance that manages all the data and batch preparation
    * num_epochs: The number of epochs for which the discriminator is trained in this sleep phase
    * use_cuda: cuda flag
    """

    # Make sure that generator gradient is off!
    print 'Train Sentiment Discriminator'

    # generated_probabilities, seed_c = cgn_model.sample_generator_for_learning(data_handler, config, use_cuda)
    # print type(generated_probabilities), generated_probabilities.size(), type(seed_c), seed_c.size()
    
    # Initialize discriminator if needed
    # cgn_model.initialize_discriminator()
    
    sentiment_disc_train_step = cgn_model.discriminator_sentiment_trainer(data_handler, use_cuda)
    
    step = 0
    num_train_batches = data_handler.num_train_sentiment_batches

    for epoch in range(0, num_epochs):

        epoch_loss = 0

        for batch_index in range(num_train_batches):

            # returns tensor with total_batch_loss
            total_batch_loss = sentiment_disc_train_step(step, batch_index)

            loss_val = total_batch_loss.data[0]
            epoch_loss += loss_val
                        
            if step % 50 == 0:
                
                print('\n')
                print ('------------------------')
                print ('Epoch: %d Batch_Index: %d' % (epoch, batch_index))
                print ('Total Training Steps: %d' % step)
                print ('Batch Loss: %f' % loss_val)

            step += 1

        epoch += 1
        print('\n')
        print ('------------------------')
        print ('Epoch: %d' % epoch)
        print ('Total Epoch Loss: %f' % epoch_loss)


def main():

    parser = argparse.ArgumentParser(description='Controlled_Generation_Sentence')

    parser.add_argument('--rvae-initial-iterations', type=int, default=120000)
    parser.add_argument('--discriminator-epochs', type=int, default=1)
    parser.add_argument('--generator-iterations', type=int, default=1000)
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

    summary_writer = SummaryWriter(ROOT_DIR+'snapshot/run_initial/')
    
    if train_initial_rvae:

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

    else:
        cgn_model.load_state_dict(t.load(args.preload_initial_rvae))
        #data_handler.load_discriminator(args.sentiment_discriminator_train_file)
        #train_sentiment_discriminator(cgn_model, config, data_handler, args.discriminator_epochs, args.use_cuda)
        train_encoder_decoder_wake_phase(cgn_model, config, data_handler, args.discriminator_epochs, args.use_cuda, args.dropout)

    summary_writer.export_scalars_to_json("./all_scalars.json")
    summary_writer.close()

    
if __name__ == '__main__':
    main()
