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

import torchvision
from tensorboardX import SummaryWriter

ROOT_DIR = '/data1/nikhil/cgn_train_data/'

# % Validation check to determine the switch
VALIDATION_DIFF = 5

def train_encoder_decoder(cgn_model, config, data_handler, num_epochs, use_cuda, dropout, summary_writer, total_steps, valid_prev_loss):

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

    batch_size = 500
    discriminator_forward = cgn_model.discriminator_forward_function (data_handler, use_cuda)
        
    num_line = (data_handler.gen_batch_loader.num_lines[0])
    num_batches = num_line / batch_size
    
    c = []
    for batch_index in range(num_batches+1):
        start_index = batch_index * batch_size
        if (batch_index < num_batches):
            c_batch = discriminator_forward(start_index, batch_size)
        elif (num_line % batch_size != 0):
            batch_size = num_line % batch_size
            c_batch = discriminator_forward(start_index, batch_size)
        else:
            break
            
        c.append(c_batch)
        if batch_index % 100 == 0:
            print 'Batch: %d/%d Start_Index: %d Batch_Size:%d' % (batch_index, num_batches, start_index, batch_size) 

    c = t.cat(c, dim=0)
    c = c.view(-1)
        
    #----------------------Now Train Encoder-Generator--------------------------------

    # Use a smaller batch_size
    # num_line = data_handler.gen_batch_loader.num_lines[0]
    train_line = data_handler.gen_batch_loader.train_lines
    train_line = train_line - train_line % data_handler.batch_size
    num_batches = train_line / data_handler.batch_size

    cgn_model.prep_enc_gen_training()
    train_enc_gen = cgn_model.train_encoder_generator(data_handler)
    valid_enc_gen = cgn_model.valid_encoder_generator(data_handler)

    batch_index = 0
    while True:
        # Train till current_valid_loss <= prev_valid_loss

        # Begin Training
        batch_index %= num_batches
        start_index = batch_index*data_handler.batch_size

        # total_steps is used for kld linear annealing
        cross_entropy, kld, generated_recon_loss, z_loss, c_loss, kld_coeff = train_enc_gen(start_index,
                                                                                                data_handler.batch_size,
                                                                                                use_cuda, dropout,
                                                                                                c_target=c, global_step=total_steps)

        ce_loss_val = cross_entropy.data.cpu()[0]
        kld_val = kld.data.cpu()[0]
        gen_loss_val = generated_recon_loss.data.cpu()[0]
        vae_loss_val = kld_val*kld_coeff+ce_loss_val
        z_loss_val = z_loss.data.cpu()[0]
        c_loss_val = c_loss.data.cpu()[0]
            
        if total_steps % 100 == 0:
                
            print ('\n')
            print ('-----------Training-------------')
            print ('Encoder-Generator Data')
            print ('Batch_Index: %d/%d' % (batch_index, num_batches))
            print ('Enc-Gen Training Step: %d' % total_steps)
            print ('Encoder Loss (VAE Loss): %f' % vae_loss_val)
            print ('Generator Loss (VAE_Loss+ lambda_z*z_recon_loss + lambda_c*c_recon_loss): %f' % (vae_loss_val+gen_loss_val))
            print ('Generator z Loss: %f' % z_loss_val)
            print ('Generator c Loss: %f' % c_loss_val)
            print ('Cross Entropy: %f' % ce_loss_val)
            print ('KLD: %f' % kld_val)
                
                
            summary_writer.add_scalar('train_enc_gen/encoder_loss', vae_loss_val, total_steps)
            summary_writer.add_scalar('train_enc_gen/generator_loss', (vae_loss_val+gen_loss_val), total_steps)
            summary_writer.add_scalar('train_enc_gen/generator_z_loss', z_loss_val, total_steps)
            summary_writer.add_scalar('train_enc_gen/generator_c_loss', c_loss_val, total_steps)
            summary_writer.add_scalar('train_enc_gen/cross_entropy', ce_loss_val, total_steps)
            summary_writer.add_scalar('train_enc_gen/kld', kld_val, total_steps)
            summary_writer.add_scalar('train_enc_gen/kld_coeff', kld_coeff)


        #------Train step completed-------

        #------Do Validation-----------

        if total_steps % 200 == 0:

            # do validation
            
            # Enc-Gen Validation mode
            cgn_model.prep_enc_gen_validation()
                                
            valid_ce_val = 0
            valid_kld_val = 0
            valid_gen_loss_val = 0
            valid_z_loss_val = 0
            valid_c_loss_val = 0

            num_valid_iterations = data_handler.gen_batch_loader.val_lines / data_handler.batch_size
            valid_index = 0

            for valid_step in range(num_valid_iterations):
                    
                valid_index = valid_step*data_handler.batch_size
                    
                # total_steps is used for kld annealing
                valid_cross_entropy, valid_kld, valid_generated_recon_loss, valid_z_loss, valid_c_loss, valid_kld_coeff = valid_enc_gen(valid_index,
                                                                                                                                        data_handler.batch_size,
                                                                                                                                        use_cuda, 0.0,
                                                                                                                                        c_target=c, global_step=total_steps)
            
                valid_ce_val += valid_cross_entropy.data.cpu()[0]
                valid_kld_val += valid_kld.data.cpu()[0]
                valid_gen_loss_val += valid_generated_recon_loss.data.cpu()[0]
                valid_z_loss_val += valid_z_loss.data.cpu()[0]
                valid_c_loss_val += valid_c_loss.data.cpu()[0]

            valid_ce_val /= num_valid_iterations
            valid_kld_val /= num_valid_iterations
            valid_gen_loss_val /= num_valid_iterations
            valid_z_loss_val /= num_valid_iterations
            valid_c_loss_val /= num_valid_iterations
            valid_vae_loss_val = valid_kld_val*valid_kld_coeff+valid_ce_val

            print ('\n')
            print ('----------Validation--------------')
            print ('Encoder-Generator Data')
            print ('Total Valid Batches%d' % (num_valid_iterations))
            print ('Enc-Gen Current Step: %d' % total_steps)
            print ('Encoder Loss (VAE Loss): %f' % valid_vae_loss_val)
            print ('Generator Loss (VAE_Loss+ lambda_z*z_recon_loss + lambda_c*c_recon_loss): %f' % (valid_vae_loss_val+valid_gen_loss_val))
            print ('Generator z Loss: %f' % valid_z_loss_val)
            print ('Generator c Loss: %f' % valid_c_loss_val)
            print ('Cross Entropy: %f' % valid_ce_val)
            print ('KLD: %f' % valid_kld_val)
                
            summary_writer.add_scalar('valid_enc_gen/encoder_loss', valid_vae_loss_val, total_steps)
            summary_writer.add_scalar('valid_enc_gen/generator_loss', (valid_vae_loss_val+valid_gen_loss_val), total_steps)
            summary_writer.add_scalar('valid_enc_gen/generator_z_loss', valid_z_loss_val, total_steps)
            summary_writer.add_scalar('valid_enc_gen/generator_c_loss', valid_c_loss_val, total_steps)
            summary_writer.add_scalar('valid_enc_gen/cross_entropy', valid_ce_val, total_steps)
            summary_writer.add_scalar('valid_enc_gen/kld', valid_kld_val, total_steps)
            summary_writer.add_scalar('valid_enc_gen/kld_coef', valid_kld_coeff, total_steps)

            valid_current_loss = valid_vae_loss_val
                
            # Back to training mode
            cgn_model.prep_enc_gen_training()

            # Check validation loss
            if (valid_current_loss > valid_prev_loss): 
                break
            else:
                valid_prev_loss = valid_current_loss
                
        total_steps += 1
        batch_index += 1

    return total_steps, valid_prev_loss

def train_sentiment_discriminator(cgn_model, config, data_handler, num_epochs, use_cuda, summary_writer, total_training_steps, valid_prev_loss):

    """
    Trains the discriminator using both the labeled data and unlabedled data (from the generator). 

    Parameters:
    * cgn_model: A cgn_model with pretrained generator and encoder weights. (See main() flags).
    * data_handler: data_handler instance that manages all the data and batch preparation
    * num_epochs: The number of epochs for which the discriminator is trained in this sleep phase
    * use_cuda: cuda flag
    * valid_disc_prev_loss: 
    """

    print 'Training Sentiment Discriminator. Total Gen-Disc Steps: %d' % total_training_steps         
    num_train_batches = data_handler.num_train_sentiment_batches
    num_valid_batches = data_handler.num_dev_sentiment_batches
    
    number_of_gen_samples = data_handler.batch_size # Number of samples to generate
    gen_samples_batch_size = 10  # Generate batch_size samples in 1 call. Increasing batch_size here will increase memory consumption proportional to O(beam_size)

    cgn_model.prep_disc_training()
    sentiment_disc_train_step = cgn_model.discriminator_sentiment_trainer(data_handler, use_cuda)
    sentiment_disc_eval_step = cgn_model.discriminator_sentiment_valid(data_handler, use_cuda)

    batch_index = 0
    while True:

        # Train till valid_current_loss <= valid_prev_loss
        batch_index %= num_train_batches
        
        generated_samples, generated_c = [], []
        for ith_sample in range(number_of_gen_samples/gen_samples_batch_size):
            gen_sample, gen_c = cgn_model.sample_generator_for_learning(data_handler, config, use_cuda, sample=gen_samples_batch_size)
            generated_samples.extend(gen_sample)
            generated_c.extend(gen_c)

        generated_c = t.stack(generated_c, dim=0)
        total_loss, batch_train_ce_loss, generated_loss, generated_loss_ce, emperical_shannon_entropy = sentiment_disc_train_step(generated_samples, generated_c, batch_index)

        loss_val = total_loss.data.cpu()[0]
        train_ce_loss_val = batch_train_ce_loss.data.cpu()[0]
        generated_loss_val = generated_loss.data.cpu()[0]
        generated_loss_ce_val = generated_loss_ce.data.cpu()[0]
        ese_loss_val = emperical_shannon_entropy.data.cpu()[0]
                        
        if total_training_steps % 5 == 0:
                
            print('\n')
            print ('------------------------')
            print ('Discriminator Training')
            print ('Batch: %d/%d' % (batch_index, num_train_batches))
            print ('Total Training Steps: %d' % total_training_steps)
            print ('Batch loss: %f' % loss_val)
            print ('Labeled cross entropy: %f' % train_ce_loss_val)
            print ('Generated cross entropy: %f' % generated_loss_ce_val)
            print ('Generated emperical_shannon_entropy: %f' % ese_loss_val)
            print ('Loss from generated data as input (generated_ce + beta*emperical shannon entropy): %f' % generated_loss_val)
                
            summary_writer.add_scalar('train_discriminator/total_loss', loss_val, total_training_steps)
            summary_writer.add_scalar('train_discriminator/labeled_cross_entropy', train_ce_loss_val, total_training_steps)
            summary_writer.add_scalar('train_discriminator/generated_total_loss', generated_loss_val, total_training_steps)
            summary_writer.add_scalar('train_discriminator/generated_cross_entropy', generated_loss_ce_val, total_training_steps)
            summary_writer.add_scalar('train_discriminator/generated_emperical_shannon', ese_loss_val, total_training_steps)
        
        # -------Training step completed-------
        # ------------Do Validation------------
        if total_training_steps % 10 == 0:
            # Do Validation
            
            # Discriminator validation mode
            cgn_model.prep_disc_validation()

            valid_ce_loss_val = 0
            for valid_index in range(num_valid_batches):
                valid_ce = sentiment_disc_eval_step(valid_index)
                valid_ce_loss_val += valid_ce.data.cpu()[0]

            valid_ce_loss_val /= num_valid_batches
                
            print('\n')
            print ('------------------------')
            print ('Discriminator Validation')
            print ('Batch: %d/%d' % (batch_index, num_train_batches))
            print ('Total Training Steps: %d' % total_training_steps)
            print ('Validation Loss: %f' % valid_ce_loss_val)
                                
            summary_writer.add_scalar('valid_discriminator/labeled_cross_entropy', valid_ce_loss_val, total_training_steps)

            valid_current_loss = valid_ce_loss_val

            if (valid_current_loss > valid_prev_loss):
                break
            else:
                valid_prev_loss = valid_current_loss
                
            # Discriminator training mode back on
            cgn_model.prep_disc_training()

        # ------Validation completed------
        
        total_training_steps += 1
        batch_index += 1
    
    # step helps keep track of generator-discriminator alternating iterations
    return total_training_steps, valid_prev_loss

def main():

    parser = argparse.ArgumentParser(description='Controlled_Generation_Sentence')

    parser.add_argument('--rvae-initial-iterations', type=int, default=120000, help="initial rvae training steps")
    parser.add_argument('--discriminator-epochs', type=int, default=10, help="num epochs for which disc is to be trained while alternating")
    parser.add_argument('--generator-epochs', type=int, default=5, help="num epochs for which gener is to be trained while alternating")
    parser.add_argument('--total-alternating-iterations', type=int, default=1000, help="number of alternating iterations")
    parser.add_argument('--batch-size', type=int, default=50)
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
    parser.add_argument('--lambda-c', type=float, default=0.1, help='C Reconstruction Generator Loss Coeff (Default=0.1)')
    parser.add_argument('--beta', type=float, default=1, help='Normalizing Coeff for entropy')
    args = parser.parse_args()

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
                ce_loss_val = cross_entropy.data.cpu()[0]
                kld_loss_val = kld.data.cpu()[0]
                total_loss_val = total_loss.data.cpu()[0]

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
                    
                    valid_ce_val += cross_entropy.data.cpu()[0]
                    valid_kld_val += kld.data.cpu()[0]
                    valid_total_loss_val += total_loss.data.cpu()[0]

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
                
            if (iteration+1) % 20000 == 0:
                t.save(cgn_model.state_dict(), os.path.join(args.save_model_dir, ('initial_rvae_updated_%d'%(iteration+1))))

    elif (args.load_cgn) :

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

        summary_dir = ROOT_DIR+'snapshot/train_cgn_logs/'
        summary_writer = SummaryWriter(summary_dir)
        #summary_writer = SummaryWriter(ROOT_DIR+'snapshot/run_random_logs/')
        
        # Load pretrained
        cgn_model.load_state_dict(t.load(args.preload_initial_rvae))

        # Load discriminator labelled data
        data_handler.load_discriminator(args.sentiment_discriminator_train_file)

        total_discriminator_steps = 0
        total_generator_steps = 0

        valid_enc_gen_prev_loss = float('inf')
        valid_disc_prev_loss = float('inf')
        
        for i in range(args.total_alternating_iterations):
            
            total_discriminator_steps, valid_disc_prev_loss = train_sentiment_discriminator(cgn_model, config, data_handler, args.discriminator_epochs, \
                                                                                            args.use_cuda, summary_writer, total_discriminator_steps, \
                                                                                            valid_disc_prev_loss)
            
            total_generator_steps, valid_enc_gen_prev_loss = train_encoder_decoder(cgn_model, config, data_handler, args.generator_epochs, \
                                                                                   args.use_cuda, args.dropout, summary_writer, \
                                                                                   total_generator_steps, valid_enc_gen_prev_loss)
            

            if (i) % 2 == 0:
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
