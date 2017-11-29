import torch as t
import numpy as np
import os

from model.cgn import Controlled_Generation_Sentence

def generate_samples_for_disc_training(cgn_model, data_handler, config, use_cuda, number_of_gen_samples, gen_samples_batch_size):

    generated_samples, generated_c = [], []
    for ith_sample in range(number_of_gen_samples/gen_samples_batch_size):
        gen_sample, gen_c = cgn_model.sample_generator_for_learning(data_handler, config, use_cuda, sample=gen_samples_batch_size)
        generated_samples.extend(gen_sample)
        generated_c.extend(gen_c)

    generated_c = t.stack(generated_c, dim=0)

    return generated_samples, generated_c

def train_complete_model (cgn_model, data_handler, config, use_cuda, dropout, summary_writer, save_model_dir, model_name):

    num_steps = 2500000

    # Set everything to train mode
    cgn_model.discriminator_mode(train_mode=True)
    cgn_model.encoder_mode(train_mode=True)
    cgn_model.generator_mode(train_mode=True)
    
    # sentiment disc train/valid
    sentiment_disc_train_step = cgn_model.discriminator_sentiment_trainer(data_handler, use_cuda)
    sentiment_disc_eval_step = cgn_model.discriminator_sentiment_valid(data_handler, use_cuda, return_accuracy=True)

    # train/valid enc gen steps
    train_enc_gen = cgn_model.train_encoder_generator(summary_writer)
    valid_enc_gen = cgn_model.valid_encoder_generator()

    train_disc_batches = data_handler.num_train_sentiment_batches
    valid_disc_batches = data_handler.num_dev_sentiment_batches
    number_of_gen_samples = data_handler.batch_size # Number of samples to generate when training disc
    gen_samples_batch_size = 10  # Generate batch_size samples in 1 call. Increasing batch_size here will increase memory consumption proportional to O(beam_size)

    # Encoder-Gen Num batches
    train_gen_line = data_handler.gen_batch_loader.train_lines
    train_gen_line = train_gen_line - train_gen_line % data_handler.batch_size
    train_gen_batches = train_gen_line / data_handler.batch_size

    # Begin Alternate Training
    disc_batch_index = 0
    enc_gen_batch_index = 0
    for step in range(num_steps):

        disc_batch_index %= train_disc_batches
        enc_gen_batch_index %= train_gen_batches
        
        #------------Train discriminator------------
        cgn_model.prep_disc_training()
        generated_samples, generated_c = generate_samples_for_disc_training(cgn_model, data_handler, config, use_cuda, number_of_gen_samples, gen_samples_batch_size)
        total_loss, batch_train_ce_loss, generated_loss, generated_loss_ce, emperical_shannon_entropy = sentiment_disc_train_step(generated_samples, generated_c, disc_batch_index)

        loss_val = total_loss.data.cpu()[0]
        train_ce_loss_val = batch_train_ce_loss.data.cpu()[0]
        generated_loss_val = generated_loss.data.cpu()[0]
        generated_loss_ce_val = generated_loss_ce.data.cpu()[0]
        ese_loss_val = emperical_shannon_entropy.data.cpu()[0]
        
        if step % 5 == 0:
            disc_train_log(disc_batch_index, train_disc_batches, step, loss_val, train_ce_loss_val, generated_loss_ce_val, ese_loss_val, generated_loss_val, summary_writer)

        # Discriminator validation
        if step % 40 == 0:
            disc_validation(cgn_model, data_handler, config, use_cuda, step, valid_disc_batches, sentiment_disc_eval_step, number_of_gen_samples, summary_writer)
    
        disc_batch_index += 1

        #----- Train the encoder-generator -------

        # Get data
        enc_gen_start_index = data_handler.batch_size * enc_gen_batch_index
        enc_gen_input = data_handler.gen_batch_loader.next_batch(data_handler.batch_size, 'train', enc_gen_start_index)

        # Get c for enc-gen batch
        cgn_model.discriminator_mode(train_mode=False)
        c_batch = cgn_model.discriminator_forward(enc_gen_input, data_handler.batch_size, use_cuda)

        # Prepare enc-gen training.
        cgn_model.prep_enc_gen_training()
                
        # total_steps is used for kld linear annealing
        cross_entropy, kld, z_loss, c_loss, kld_coef, temp_coef, total_encoder_loss, total_generator_loss = train_enc_gen(enc_gen_input, data_handler.batch_size,
                                                                                                                           use_cuda, dropout,
                                                                                                                           c_batch, step, calc_z_loss=False)
        
        ce_loss_val = cross_entropy.data.cpu()[0]
        kld_val = kld.data.cpu()[0]
        total_generator_loss_val = total_generator_loss.data.cpu()[0]
        total_encoder_loss_val = total_encoder_loss.data.cpu()[0]
        z_loss_val = z_loss.data.cpu()[0]
        c_loss_val = c_loss.data.cpu()[0]

        if step % 50 == 0:
            enc_gen_train_log (step, total_encoder_loss_val, total_generator_loss_val, z_loss_val, c_loss_val, ce_loss_val, kld_val, kld_coef, temp_coef, summary_writer)

        if step % 100 == 0:
            enc_gen_validation (cgn_model, data_handler, use_cuda, step, valid_enc_gen, summary_writer)

        enc_gen_batch_index += 1

        if (step+1) % 5000 == 0:
            t.save(cgn_model.state_dict(), os.path.join(save_model_dir, 'cgn_model_experiment_'+model_name))

    
def disc_train_log(batch_index, num_train_batches, step, loss_val, train_ce_loss_val, generated_loss_ce_val, ese_loss_val, generated_loss_val, summary_writer):

    print('\n')
    print ('------------------------')
    print ('Discriminator Training')
    print ('Batch: %d/%d' % (batch_index, num_train_batches))
    print ('Total Training Step: %d' % step)
    print ('Batch loss: %f' % loss_val)
    print ('Labeled cross entropy: %f' % train_ce_loss_val)
    print ('Generated cross entropy: %f' % generated_loss_ce_val)
    print ('Generated emperical_shannon_entropy: %f' % ese_loss_val)
    print ('Loss from generated data as input (generated_ce + beta*emperical shannon entropy): %f' % generated_loss_val)
    
    summary_writer.add_scalar('train_discriminator/total_loss', loss_val, step)
    summary_writer.add_scalar('train_discriminator/labeled_cross_entropy', train_ce_loss_val, step)
    summary_writer.add_scalar('train_discriminator/generated_total_loss', generated_loss_val, step)
    summary_writer.add_scalar('train_discriminator/generated_cross_entropy', generated_loss_ce_val, step)
    summary_writer.add_scalar('train_discriminator/generated_emperical_shannon', ese_loss_val, step)

def enc_gen_train_log (step, total_encoder_loss_val, total_generator_loss_val, z_loss_val, c_loss_val, ce_loss_val, kld_val, kld_coef, temp_coef, summary_writer):
                
    print ('\n')
    print ('-----------Training-------------')
    print ('Encoder-Generator Data')
    print ('Enc-Gen Training Step: %d' % step)
    print ('Encoder Loss (VAE Loss): %f' % total_encoder_loss_val)
    print ('Generator Loss (VAE_Loss+ lambda_z*z_recon_loss + lambda_c*c_recon_loss): %f' % (total_generator_loss_val))
    print ('Generator z Loss: %f' % z_loss_val)
    print ('Generator c Loss: %f' % c_loss_val)
    print ('Cross Entropy: %f' % ce_loss_val)
    print ('KLD: %f' % kld_val)
    
    summary_writer.add_scalar('train_enc_gen/encoder_loss', total_encoder_loss_val, step)
    summary_writer.add_scalar('train_enc_gen/generator_loss', total_generator_loss_val, step)
    summary_writer.add_scalar('train_enc_gen/generator_z_loss', z_loss_val, step)
    summary_writer.add_scalar('train_enc_gen/generator_c_loss', c_loss_val, step)
    summary_writer.add_scalar('train_enc_gen/cross_entropy', ce_loss_val, step)
    summary_writer.add_scalar('train_enc_gen/kld', kld_val, step)
    summary_writer.add_scalar('train_enc_gen/kld_coeff', kld_coef, step)
    summary_writer.add_scalar('train_enc_gen/temp_coef', temp_coef, step)

def enc_gen_validation (cgn_model, data_handler, use_cuda, step, valid_enc_gen, summary_writer):

    cgn_model.prep_enc_gen_validation()
                                
    valid_ce_val = 0
    valid_kld_val = 0
    valid_gen_loss_val = 0
    valid_z_loss_val = 0
    valid_c_loss_val = 0
    valid_total_enc_loss_val = 0
    valid_total_gen_loss_val = 0

    # Increasing batch_size here 
    batch_size = 500
    num_valid_iterations = data_handler.gen_batch_loader.val_lines / batch_size
    valid_index = 0

    for valid_step in range(num_valid_iterations):
         
        # Get data
        valid_index = valid_step*batch_size
        enc_gen_input = data_handler.gen_batch_loader.next_batch(batch_size, 'valid', valid_index)

        # Get c for enc-gen batch
        c = cgn_model.discriminator_forward(enc_gen_input, batch_size, use_cuda)
        
        # total_steps is used for kld annealing
        valid_cross_entropy, valid_kld, valid_z_loss, valid_c_loss, \
            valid_kld_coeff, valid_temp_coeff, valid_total_enc_loss, valid_total_gen_loss = valid_enc_gen(enc_gen_input,
                                                                                                          batch_size,
                                                                                                          use_cuda, c_target=c,
                                                                                                          global_step=step,
                                                                                                          calc_z_loss=False)
            
        valid_ce_val += valid_cross_entropy.data.cpu()[0]
        valid_kld_val += valid_kld.data.cpu()[0]
        valid_total_enc_loss_val += valid_total_enc_loss.data.cpu()[0]
        valid_total_gen_loss_val += valid_total_gen_loss.data.cpu()[0]
        valid_z_loss_val += valid_z_loss.data.cpu()[0]
        valid_c_loss_val += valid_c_loss.data.cpu()[0]

    valid_ce_val /= num_valid_iterations
    valid_kld_val /= num_valid_iterations
    valid_total_gen_loss_val /= num_valid_iterations
    valid_total_enc_loss_val /= num_valid_iterations
    valid_z_loss_val /= num_valid_iterations
    valid_c_loss_val /= num_valid_iterations
            
    print ('\n')
    print ('----------Validation--------------')
    print ('Encoder-Generator Data')
    print ('Total Valid Batches%d' % (num_valid_iterations))
    print ('Enc-Gen Current Step: %d' % step)
    print ('Encoder Loss (VAE Loss): %f' % valid_total_enc_loss_val)
    print ('Generator Loss (VAE_Loss+ lambda_z*z_recon_loss + lambda_c*c_recon_loss): %f' % (valid_total_gen_loss_val))
    print ('Generator z Loss: %f' % valid_z_loss_val)
    print ('Generator c Loss: %f' % valid_c_loss_val)
    print ('Cross Entropy: %f' % valid_ce_val)
    print ('KLD: %f' % valid_kld_val)
                
    summary_writer.add_scalar('valid_enc_gen/encoder_loss', valid_total_enc_loss_val, step)
    summary_writer.add_scalar('valid_enc_gen/generator_loss', valid_total_gen_loss_val, step)
    summary_writer.add_scalar('valid_enc_gen/generator_z_loss', valid_z_loss_val, step)
    summary_writer.add_scalar('valid_enc_gen/generator_c_loss', valid_c_loss_val, step)
    summary_writer.add_scalar('valid_enc_gen/cross_entropy', valid_ce_val, step)
    summary_writer.add_scalar('valid_enc_gen/kld', valid_kld_val, step)
    summary_writer.add_scalar('valid_enc_gen/kld_coef', valid_kld_coeff, step)
    summary_writer.add_scalar('valid_enc_gen/temp_coef', valid_temp_coeff, step) 
    
    # Back to training mode
    cgn_model.prep_enc_gen_training()
        
def disc_validation(cgn_model, data_handler, config, use_cuda, step, num_valid_batches, sentiment_disc_eval_step, number_of_gen_samples, summary_writer):

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
    print ('Step: %d' % step)
    print ('Labelled Validation Loss: %f' % valid_ce_loss_val)
    print ('Generated Validation Loss: %f' % generated_ce_loss_val)
    print ('Validation Accuracy: %f' % valid_accuracy)
    print ('Generated Validation Accuracy: %f' % generated_accuracy)
    
    summary_writer.add_scalar('valid_discriminator/labeled_cross_entropy', valid_ce_loss_val, step)
    summary_writer.add_scalar('valid_discriminator/labeled_accuracy', valid_accuracy, step)
    summary_writer.add_scalar('valid_discriminator/generated_cross_entropy', generated_ce_loss_val, step)
    summary_writer.add_scalar('valid_discriminator/generated_accuracy', generated_accuracy, step)
    
    # Discriminator training mode back on
    cgn_model.prep_disc_training()
    
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
        
        if total_training_steps % 40 == 0:
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

    print ('\n')
    print ('Getting c using current state of discriminator')
    batch_size = 500
            
    num_line = (data_handler.gen_batch_loader.num_lines[0])
    num_batches = num_line / batch_size

    c = []
    for batch_index in range(num_batches+1):
        start_index = batch_index * batch_size
        if (batch_index < num_batches):
            c_batch = cgn_model.discriminator_forward(data_handler,  start_index, batch_size, use_cuda)
        elif (num_line % batch_size != 0):
            batch_size = num_line % batch_size
            c_batch = cgn_model.discriminator_forward(data_handler, start_index, batch_size, use_cuda)
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
    train_enc_gen = cgn_model.train_encoder_generator(data_handler, summary_writer)
    valid_enc_gen = cgn_model.valid_encoder_generator(data_handler)

    batch_index = 0
    current_iteration_steps = 0

    for epoch in range(num_epochs):


        for batch_index in range(num_batches):

            start_index = batch_index * data_handler.batch_size
            #current_iteration_steps += 1
    
            # total_steps is used for kld linear annealing
            cross_entropy, kld, z_loss, c_loss, kld_coeff, temp_coef, total_encoder_loss, total_generator_loss = train_enc_gen(start_index,
                                                                                                                           data_handler.batch_size,
                                                                                                                           use_cuda, dropout,
                                                                                                                           c_target=c, global_step=total_steps,
                                                                                                                           calc_z_loss=False)
        
            ce_loss_val = cross_entropy.data.cpu()[0]
            kld_val = kld.data.cpu()[0]
            total_generator_loss_val = total_generator_loss.data.cpu()[0]
            total_encoder_loss_val = total_encoder_loss.data.cpu()[0]
            z_loss_val = z_loss.data.cpu()[0]
            c_loss_val = c_loss.data.cpu()[0]
            
            if total_steps % 60 == 0:
                
                print ('\n')
                print ('-----------Training-------------')
                print ('Encoder-Generator Data')
                print ('Epoch Completed: %d Batch_Index: %d/%d' % (epoch, batch_index, num_batches))
                print ('Enc-Gen Training Step: %d' % total_steps)
                print ('Encoder Loss (VAE Loss): %f' % total_encoder_loss_val)
                print ('Generator Loss (VAE_Loss+ lambda_z*z_recon_loss + lambda_c*c_recon_loss): %f' % (total_generator_loss_val))
                print ('Generator z Loss: %f' % z_loss_val)
                print ('Generator c Loss: %f' % c_loss_val)
                print ('Cross Entropy: %f' % ce_loss_val)
                print ('KLD: %f' % kld_val)
                
                
                summary_writer.add_scalar('train_enc_gen/encoder_loss', total_encoder_loss_val, total_steps)
                summary_writer.add_scalar('train_enc_gen/generator_loss', total_generator_loss_val, total_steps)
                summary_writer.add_scalar('train_enc_gen/generator_z_loss', z_loss_val, total_steps)
                summary_writer.add_scalar('train_enc_gen/generator_c_loss', c_loss_val, total_steps)
                summary_writer.add_scalar('train_enc_gen/cross_entropy', ce_loss_val, total_steps)
                summary_writer.add_scalar('train_enc_gen/kld', kld_val, total_steps)
                summary_writer.add_scalar('train_enc_gen/kld_coeff', kld_coeff, total_steps)
                summary_writer.add_scalar('train_enc_gen/temp_coef', temp_coef, total_steps)

            total_steps += 1
            #------Train step completed-------

        #------Do Validation-----------
    
        # Enc-Gen Validation mode
        cgn_model.prep_enc_gen_validation()
                                
        valid_ce_val = 0
        valid_kld_val = 0
        valid_gen_loss_val = 0
        valid_z_loss_val = 0
        valid_c_loss_val = 0
        valid_total_enc_loss_val = 0
        valid_total_gen_loss_val = 0
            
        num_valid_iterations = data_handler.gen_batch_loader.val_lines / data_handler.batch_size
        valid_index = 0

        for valid_step in range(num_valid_iterations):
                    
            valid_index = valid_step*data_handler.batch_size
                    
            # total_steps is used for kld annealing
            valid_cross_entropy, valid_kld, valid_z_loss, valid_c_loss, \
                valid_kld_coeff, valid_temp_coeff, valid_total_enc_loss, valid_total_gen_loss = \
                                                                                                valid_enc_gen(valid_index,
                                                                                                              data_handler.batch_size,
                                                                                                              use_cuda, c_target=c,
                                                                                                              global_step=total_steps,
                                                                                                              calc_z_loss=False)
            
            valid_ce_val += valid_cross_entropy.data.cpu()[0]
            valid_kld_val += valid_kld.data.cpu()[0]
            valid_total_enc_loss_val += valid_total_enc_loss.data.cpu()[0]
            valid_total_gen_loss_val += valid_total_gen_loss.data.cpu()[0]
            valid_z_loss_val += valid_z_loss.data.cpu()[0]
            valid_c_loss_val += valid_c_loss.data.cpu()[0]

        valid_ce_val /= num_valid_iterations
        valid_kld_val /= num_valid_iterations
        valid_total_gen_loss_val /= num_valid_iterations
        valid_total_enc_loss_val /= num_valid_iterations
        valid_z_loss_val /= num_valid_iterations
        valid_c_loss_val /= num_valid_iterations
            
        print ('\n')
        print ('----------Validation--------------')
        print ('Encoder-Generator Data')
        print ('Total Valid Batches%d' % (num_valid_iterations))
        print ('Enc-Gen Current Step: %d' % total_steps)
        print ('Encoder Loss (VAE Loss): %f' % valid_total_enc_loss_val)
        print ('Generator Loss (VAE_Loss+ lambda_z*z_recon_loss + lambda_c*c_recon_loss): %f' % (valid_total_gen_loss_val))
        print ('Generator z Loss: %f' % valid_z_loss_val)
        print ('Generator c Loss: %f' % valid_c_loss_val)
        print ('Cross Entropy: %f' % valid_ce_val)
        print ('KLD: %f' % valid_kld_val)
                
        summary_writer.add_scalar('valid_enc_gen/encoder_loss', valid_total_enc_loss_val, total_steps)
        summary_writer.add_scalar('valid_enc_gen/generator_loss', valid_total_gen_loss_val, total_steps)
        summary_writer.add_scalar('valid_enc_gen/generator_z_loss', valid_z_loss_val, total_steps)
        summary_writer.add_scalar('valid_enc_gen/generator_c_loss', valid_c_loss_val, total_steps)
        summary_writer.add_scalar('valid_enc_gen/cross_entropy', valid_ce_val, total_steps)
        summary_writer.add_scalar('valid_enc_gen/kld', valid_kld_val, total_steps)
        summary_writer.add_scalar('valid_enc_gen/kld_coef', valid_kld_coeff, total_steps)
        summary_writer.add_scalar('valid_enc_gen/temp_coef', valid_temp_coeff, total_steps) 
        summary_writer.add_scalar('valid_enc_gen/epoch', epoch, total_steps)
        #valid_current_loss = 2*valid_vae_loss_val + valid_gen_loss_val
                
        # Back to training mode
        cgn_model.prep_enc_gen_training()
        
    return total_steps
