import argparse
import os
import sys
import re

import numpy as np
import torch as t

from utils.data_handler import DataHandler
from utils.config import Config

from model.cgn import Controlled_Generation_Sentence

ROOT_DIR = '/data1/nikhil/sentence-corpus/'

"""

TODO

Create separate training params for generator and discriminator and then average them
This should help analyze the training progress 


"""

def train_sentiment_discriminator(cgn_model, data_handler, num_epochs):

    print 'Train Sentiment Discriminator'
    
    sentiment_disc_train_step = cgn_model.discriminator_sentiment_trainer(data_handler)
    # validate_step ?

    step = 0
    num_train_batches = data_handler.num_train_sentiment_batches

    for epoch in range(1, num_epochs):

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

    if train_initial_rvae:

        start_iteration = 0
        
        initial_train_step = cgn_model.initial_trainer(data_handler)
        # initial_validate_step ?

        start_index = 0
        ce_result = 0
        kld_result = 0

        num_line = (data_handler.gen_batch_loader.num_lines[0])
        num_line = num_line - num_line % args.batch_size
        print 'Begin Step 1. Initial VAE Training'

        for iteration in range(start_iteration, args.rvae_initial_iterations):

            start_index = (start_index+args.batch_size)%num_line
            cross_entropy, kld, coef = initial_train_step(iteration, args.batch_size, args.use_cuda, args.dropout, start_index)

            if iteration % 50 == 0:
                ce_loss_val = cross_entropy.data.cpu().numpy()[0]
                kld_loss_val = kld.data.cpu().numpy()[0]

                print('\n')
                print ('------------------------')
                print ('Iteration: %d'%iteration)
                print ('Cross entropy: %f'%ce_loss_val)
                print ('KLD: %f'%kld_loss_val)
                print ('KLD Coef: %f'%coef)
                print ('Total Loss: %f'%(ce_loss_val+kld_loss_val))

        t.save(cgn_model.state_dict(), os.path.join(args.save_model_dir, 'initial_rvae'))

    elif (args.sample_generator) :

        # load a pretrained model if needed
        cgn_model.load_state_dict(t.load(args.preload_initial_rvae))
        print 'Initial Model Loaded'
        cgn_model.sample(data_handler, config, args.use_cuda)

    else:
        
        # Discriminator Training Data
        data_handler.load_discriminator(args.sentiment_discriminator_train_file)    
        train_sentiment_discriminator(cgn_model, data_handler, args.discriminator_epochs)
        
    sys.exit(0)
    
    
if __name__ == '__main__':
    main()
