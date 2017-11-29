__author__ = "Nikhil Mehta"
__copyright__ = "--"
#---------------------------

import argparse
import os, errno
import sys
import re

import numpy as np
import torch as t

from utils.data_handler import DataHandler
from utils.config import Config

from model.cgn import Controlled_Generation_Sentence
from train_functions import train_complete_model, generate_samples_for_disc_training

import torchvision
from tensorboardX import SummaryWriter

ROOT_DIR = '/data1/nikhil/cgn_train_data_big/'                                                                                              
        
def main():

    parser = argparse.ArgumentParser(description='Controlled_Generation_Sentence')

    parser.add_argument('--rvae-initial-iterations', type=int, default=360000, help="initial rvae training steps")
    parser.add_argument('--discriminator-epochs', type=int, default=10, help="num epochs for which disc is to be trained while alternating")
    parser.add_argument('--generator-epochs', type=int, default=5, help="num epochs for which gener is to be trained while alternating")
    parser.add_argument('--total-alternating-iterations', type=int, default=1000, help="number of alternating iterations")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--sample-generator', type=bool, default=False)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--generator-train-file', type=str, default=os.path.join(ROOT_DIR, 'raw_train_generator.pkl'))
    parser.add_argument('--embedding-path', type=str, default=os.path.join(ROOT_DIR, 'word_embeddings.npy'))
    parser.add_argument('--preload-initial-rvae', type=str, default=None)
    parser.add_argument('--load-cgn', type=str, default=None)
    parser.add_argument('--save-model-dir', type=str, default=os.path.join(ROOT_DIR, 'snapshot/'))
    parser.add_argument('--words-vocab-path', type=str, default=os.path.join(ROOT_DIR, 'words_vocab.pkl'))
    parser.add_argument('--chars-vocab-path', type=str, default=os.path.join(ROOT_DIR, 'characters_vocab.pkl'))
    parser.add_argument('--sentiment-discriminator-train-file', type=str, default=os.path.join(ROOT_DIR, 'raw_train_sentiment_discriminator.pkl'))
    parser.add_argument('--train-cgn-model', type=bool, default=False)
    parser.add_argument('--lambda-z', type=float, default=0.1, help='Z Reconstruction Generator Loss Coeff (Default=0.1)')
    parser.add_argument('--lambda-u', type=float, default=0.1, help='C Reconstruction for Generated Data (Sleep phase) Discriminator Loss Coeff (Default=0.1)')
    parser.add_argument('--lambda-c', type=float, default=7.9, help='C Reconstruction Generator Loss Coeff (Default=10)')
    parser.add_argument('--beta', type=float, default=1, help='Normalizing Coeff for entropy used in Discriminator Loss')
    parser.add_argument('--logdir', type=str, help='Relative path of log dir')
    parser.add_argument('--compute-accuracy', type=bool, default=False, help='Compute Discriminator Accuracy')
    
    args = parser.parse_args()

    if not args.logdir and args.train_cgn_model:
        print 'Please enter the log-dir name'
        sys.exit()
    
    if not os.path.exists(args.embedding_path):
        raise IOError("Word Embedding file cannot be read")

    if not os.path.exists(args.words_vocab_path):
        raise IOError("Words Vocabulary file cannot be read")

    if not os.path.exists(args.chars_vocab_path):
        raise IOError("Characters Vocabulary file cannot be read")

    if not os.path.exists(args.generator_train_file):
        raise IOError("Generator Training file cannot be read")

    if not os.path.exists(args.sentiment_discriminator_train_file):
        raise IOError("Sentiment Training file cannot be read")

    if not os.path.exists(args.save_model_dir):
        try:
            os.makedirs(args.save_model_dir)
        except OSError as e:
            raise OSError('Directory can\'t be created')
    
    vocab_files = [args.words_vocab_path, args.chars_vocab_path]

    train_initial_rvae = (args.preload_initial_rvae == None and args.load_cgn == None)

    if args.sample_generator:
        data_handler = DataHandler(vocab_files, None, args.batch_size)
    else:
        data_handler = DataHandler(vocab_files, args.generator_train_file, args.batch_size)    
    
    config = Config(data_handler.gen_batch_loader.max_word_len, data_handler.gen_batch_loader.max_seq_len, data_handler.gen_batch_loader.words_vocab_size, data_handler.gen_batch_loader.chars_vocab_size, args.learning_rate, args.lambda_c, args.lambda_z, args.lambda_u, args.beta)
    
    cgn_model = Controlled_Generation_Sentence(config, args.embedding_path)

    if args.use_cuda:
        cgn_model = cgn_model.cuda()

        
    if train_initial_rvae:

        summary_dir = ROOT_DIR+'snapshot/run_initial_update/'
        summary_writer = SummaryWriter(summary_dir)

        start_iteration = 0

        initial_train_step = cgn_model.initial_rvae_trainer(data_handler)
        initial_valid_step = cgn_model.initial_rvae_valid(data_handler)

        start_index = 0
        ce_result = 0
        kld_result = 0

        train_lines = data_handler.gen_batch_loader.train_lines
        train_lines = train_lines - train_lines % args.batch_size
        #num_line = (data_handler.gen_batch_loader.num_lines[0])
        #num_line = num_line - num_line % args.batch_size
        print 'Begin Step 1. Initial VAE Training'

        # Sets train mode for enc-gen
        cgn_model.prep_enc_gen_training()
        
        for iteration in range(start_iteration, args.rvae_initial_iterations):

            start_index = (start_index+args.batch_size)%train_lines
            cross_entropy, kld, coef, total_loss = initial_train_step(iteration, args.batch_size, args.use_cuda, args.dropout, start_index)

            if iteration % 50 == 0:
                ce_loss_val = cross_entropy.cpu()[0]
                kld_loss_val = kld.cpu()[0]
                total_loss_val = total_loss.cpu()[0]

                print('\n')
                print ('-----------Training-------------')
                print ('Iteration: %d'%iteration)
                print ('Cross entropy: %f'%ce_loss_val)
                print ('KLD: %f'%kld_loss_val)
                print ('KLD Coef: %f' % coef)
                print ('Total Loss: %f'%(total_loss_val))
                
                summary_writer.add_scalar('train_initial_rvae/kld_coef', coef, iteration)
                summary_writer.add_scalar('train_initial_rvae/kld', kld_loss_val, iteration)
                summary_writer.add_scalar('train_initial_rvae/cross_entropy', ce_loss_val, iteration)                
                summary_writer.add_scalar('train_initial_rvae/total_loss', (total_loss_val), iteration)


            if iteration % 500 == 0:
                # do validation

                # Validation mode. Calls .eval() on modules
                cgn_model.prep_enc_gen_validation()
                
                valid_ce_val = 0
                valid_kld_val = 0
                valid_total_loss_val = 0

                num_valid_iterations = data_handler.gen_batch_loader.val_lines / args.batch_size
                valid_index = 0
                
                for valid_step in range(num_valid_iterations):
                    
                    valid_index = valid_step*args.batch_size
                    cross_entropy, kld, coef, total_loss = initial_valid_step(iteration, args.batch_size, args.use_cuda, 1, valid_index)
                    
                    valid_ce_val += cross_entropy.cpu()[0]
                    valid_kld_val += kld.cpu()[0]
                    valid_total_loss_val += total_loss.cpu()[0]

                valid_ce_val /= num_valid_iterations
                valid_kld_val /= num_valid_iterations
                valid_total_loss_val /= num_valid_iterations
                
                print('\n')
                print ('-----------Validation-------------')
                print ('Iteration: %d'%iteration)
                print ('Total Cross entropy: %f'%valid_ce_val)
                print ('KLD: %f'%valid_kld_val)
                print ('KLD Coef: %f' % coef)
                print ('Total Loss: %f'%(valid_total_loss_val))
                    
                summary_writer.add_scalar('valid_initial_rvae/kld_coef', coef, iteration)
                summary_writer.add_scalar('valid_initial_rvae/kld', valid_kld_val, iteration)
                summary_writer.add_scalar('valid_initial_rvae/cross_entropy', valid_ce_val, iteration)                
                summary_writer.add_scalar('valid_initial_rvae/total_loss', valid_total_loss_val, iteration)

                # Back to training mode
                cgn_model.prep_enc_gen_training()
                
            if (iteration+1) % 60000 == 0:
                t.save(cgn_model.state_dict(), os.path.join(args.save_model_dir, ('initial_rvae_updated_%d'%(iteration+1))))
    
    elif (args.load_cgn and args.sample_generator) :

        # load cgn model and sample
        cgn_model.load_state_dict(t.load(args.load_cgn))
        print 'CGN Model loaded'
        cgn_model.sample(data_handler, config, args.use_cuda)
        sys.exit()
        
    elif (args.sample_generator) :

        # load a pretrained model if needed
        cgn_model.load_state_dict(t.load(args.preload_initial_rvae))
        print 'Initial Model Loaded'
        cgn_model.sample(data_handler, config, args.use_cuda)
        sys.exit()
        
    elif (args.train_cgn_model):

        summary_dir = ROOT_DIR+'snapshot/'+args.logdir
        summary_writer = SummaryWriter(summary_dir)

        print summary_dir

        # Load discriminator labelled data
        data_handler.load_discriminator(args.sentiment_discriminator_train_file)

        # Load initial rvae
        cgn_model.load_state_dict(t.load(args.preload_initial_rvae))

        # train_complete_model
        train_complete_model(cgn_model, data_handler, config, args.use_cuda, args.dropout, summary_writer, args.save_model_dir, args.logdir)

        sys.exit()
        # Remove this
        
        total_discriminator_steps = 0
        total_generator_steps = 0

        valid_enc_gen_prev_loss = float('inf')
        valid_disc_prev_loss = float('inf')
        skip_first_disc_training = False

        # Load pretrained
        if args.preload_initial_rvae:
            cgn_model.load_state_dict(t.load(args.preload_initial_rvae))
            print 'Preload initial rvae'
            total_discriminator_steps, valid_disc_prev_loss = train_sentiment_discriminator(cgn_model, config, data_handler, args.discriminator_epochs, \
                                                                                            args.use_cuda, summary_writer, total_discriminator_steps, \
                                                                                            valid_disc_prev_loss)
            t.save(cgn_model.state_dict(), os.path.join(args.save_model_dir, 'cgn_model_experiment_'+args.logdir+"_disc_pretrained"))

        elif args.load_cgn:
            cgn_model.load_state_dict(t.load(args.load_cgn))
            print 'CGN Loaded: ' + args.load_cgn

        
        for i in range(args.total_alternating_iterations):
                                            
            total_generator_steps = train_encoder_decoder(cgn_model, config, data_handler, args.generator_epochs, \
                                                                                   args.use_cuda, args.dropout, summary_writer, \
                                                                                   total_generator_steps)
            

            total_discriminator_steps, valid_disc_prev_loss = train_sentiment_discriminator(cgn_model, config, data_handler, args.discriminator_epochs, \
                                                                                            args.use_cuda, summary_writer, total_discriminator_steps, \
                                                                                            valid_disc_prev_loss)

            #if (i) % 2 == 0:
            t.save(cgn_model.state_dict(), os.path.join(args.save_model_dir, 'cgn_model_experiment_'+args.logdir))

    elif args.compute_accuracy:

        summary_writer = SummaryWriter(ROOT_DIR+'snapshot/run_random_logs/')
        if args.preload_initial_rvae:
            cgn_model.load_state_dict(t.load(args.preload_initial_rvae))
            print 'Preload initial rvae ' + args.preload_initial_rvae
        elif args.load_cgn:
            cgn_model.load_state_dict(t.load(args.load_cgn))
            print 'CGN Loaded: ' + args.load_cgn

        # Load discriminator labelled data
        data_handler.load_discriminator(args.sentiment_discriminator_train_file)
        compute_disc_accuracy(cgn_model, data_handler, args.use_cuda, config)
        
    else:

        summary_writer = SummaryWriter(ROOT_DIR+'snapshot/run_random_logs/')
        # cgn_model.load_state_dict(t.load(args.preload_initial_rvae))
        # data_handler.load_discriminator(args.sentiment_discriminator_train_file)
        # train_sentiment_discriminator(cgn_model, config, data_handler, args.discriminator_epochs, args.use_cuda)
        # train_encoder_decoder (cgn_model, config, data_handler, args.generator_epochs, args.use_cuda, args.dropout)

    summary_writer.close()

def compute_disc_accuracy(cgn_model, data_handler, use_cuda, config):

    """
    Computes the  discriminator accuracy using the validation data

    Parameters:
    * cgn_model: A cgn_model with pretrained generator and encoder weights. (See main() flags).
    * data_handler: data_handler instance that manages all the data and batch preparation
    * use_cuda: cuda flag
    """

    print 'Compute Sentiment Discriminator Accuracy'
    num_valid_batches = data_handler.num_dev_sentiment_batches
    number_of_gen_samples = data_handler.batch_size
    
    sentiment_disc_eval_step = cgn_model.discriminator_sentiment_valid(data_handler, use_cuda, return_accuracy=True)

    # Discriminator validation mode
    cgn_model.prep_disc_validation()

    valid_ce_loss_val = 0
    generated_ce_loss_val = 0
    valid_correct = 0
    generated_correct = 0
    for valid_index in range(num_valid_batches):
        # generate some data
        generated_samples, generated_c = generate_samples_for_disc_training(cgn_model, data_handler, config, use_cuda, number_of_gen_samples, 10)

        valid_ce, generated_ce, valid_corr, generated_corr = sentiment_disc_eval_step(valid_index, generated_samples, generated_c)
        valid_ce_loss_val += valid_ce.data.cpu()[0]
        generated_ce_loss_val += generated_ce.data.cpu()[0]
        valid_correct += valid_corr.data.cpu()[0]
        generated_correct += generated_corr.data.cpu()[0]

    valid_ce_loss_val /= num_valid_batches
    generated_ce_loss_val /= num_valid_batches
    valid_accuracy = valid_correct * 100/ (data_handler.batch_size*num_valid_batches)
    generated_accuracy = generated_correct * 100/ (number_of_gen_samples*num_valid_batches)
    
    print('\n')
    print ('------------------------')
    print ('Discriminator Validation')
    print ('Labelled Validation Loss: %f' % valid_ce_loss_val)
    print ('Generated Validation Loss: %f' % generated_ce_loss_val)
    print ('Validation Accuracy: %f' % valid_accuracy)
    print ('Generated Validation Accuracy: %f' % generated_accuracy)
             
if __name__ == '__main__':
    main()
