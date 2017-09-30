import argparse
import os
import sys
import re

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

def sample (rvae, parameters, batch_loader, use_cuda=True, samp=1):

    seed = t.randn([samp, parameters.latent_variable_size])
    sentences, result_score  = rvae.sample_beam_for_decoder(batch_loader, parameters.max_seq_len, seed, use_cuda, samples=samp)

    for s in sentences:
        print s
    #for r in result:
    #    sen.append(batch_loader.decode_word(r))

    

def main():

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=10000, metavar='NI',
                        help='num iterations (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=str, default=None, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')
    parser.add_argument('--train-file', type=str, default='/data1/nikhil/sentence-corpus/generator/train.txt', metavar='NS',
                        help='train file path (default: /data1/nikhil/sentence-corpus/generator/train.txt)')
    parser.add_argument('--embedding-path', type=str, default='/data1/nikhil/sentence-corpus/generator/word_embeddings.npy', metavar='EP',
                        help="ep path: (default: /data1/nikhil/sentence-corpus/generator/word_embeddings.npy)")
    parser.add_argument('--save-model', type=str, default='/data1/nikhil/sentence-corpus/generator/trained_RVAE_final', metavar='NS',
                        help='trained model save path (default: trained_RVAE)')
    parser.add_argument('--words-vocab-path', type=str, default='/data1/nikhil/sentence-corpus/generator/words_vocab.pkl', metavar='wv',
                        help="wv path: (default: /data1/nikhil/sentence-corpus/generator/words_vocab.pkl)")
    parser.add_argument('--chars-vocab-path', type=str, default='/data1/nikhil/sentence-corpus/generator/characters_vocab.pkl', metavar='cv',
                        help="cv path: (default: /data1/nikhil/sentence-corpus/generator/characters_vocab.pkl)")
    parser.add_argument('--sample-sentence', type=int, default=0, metavar='SS')
    

    args = parser.parse_args()

    if not os.path.exists(args.embedding_path):
        raise FileNotFoundError("word embeddings file was't found")
    
    idx_files = [args.words_vocab_path,
                      args.chars_vocab_path]

    ''' =================== Creating batch_loader for Encoder =========================================
    '''
    
    data_files = [args.train_file]
    data = [open(file, "r").read() for file in data_files]

    batch_loader = BatchLoader(data, idx_files)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)
    
    '''================================ RVAE object creation ===========================================
    '''

    rvae = RVAE(parameters, args.embedding_path)
    start_iteration = 0
    if args.use_trained:
        #pattern = re.compile(args.save_model + "_(\d)_1")
        match = re.search(args.save_model + r'_(\d+)', args.use_trained)
        if match:
            start_iteration = int(match.group(1))
                    
        print "Start Iteration: %d" % start_iteration
        rvae.load_state_dict(t.load(args.use_trained))
        
    if args.use_cuda:
        rvae = rvae.cuda()

    if args.sample_sentence > 0:
        sample(rvae, parameters, batch_loader, args.use_cuda, args.sample_sentence)
        sys.exit(0)
        
    '''=================================================================================================
    '''
    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    start_index = 0
    ce_result = []
    kld_result = []

    num_line = (batch_loader.num_lines[0]-args.batch_size-7) # What is 7?

    print 'Begin Training'
    for iteration in range(start_iteration, args.num_iterations):

        start_index = (start_index+args.batch_size)%num_line
        cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout, start_index)
        
        if iteration % 100 == 0:
            print('\n')

            print('----------ITERATION-----------')
            print(iteration)

            print('--------CROSS-ENTROPY---------')
            cross_entropy_loss = cross_entropy.data.cpu().numpy()[0]
            print(cross_entropy_loss)

            print('-------------KLD--------------')
            kld_loss = kld.data.cpu().numpy()[0]
            print(kld_loss)
            
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')

            print('-----------TOTAL-LOSS-----------')
            print(kld_loss + cross_entropy_loss)
            print('------------------------------')

            
        if iteration % 2000 == 0:
            print ('\n')
            print ('Sampling')

            sample(rvae, parameters, batch_loader, args.use_cuda)
            #t.save(rvae.state_dict(), args.save_model + ("_%d"%iteration))

            
    #t.save(rvae.state_dict(), args.save_model)
#     np.save('ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
#     np.save('kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))

if __name__ == "__main__":
    main()

