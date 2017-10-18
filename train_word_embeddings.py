# -*- coding: utf-8 -*-
import argparse
import sys

import cPickle as pkl
import numpy as np
import torch as t
from torch.autograd import Variable
from torch.optim import SGD

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from selfModules.neg import NEG_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('--num-iterations', type=int, default=1000000, metavar='NI',
                        help='num iterations (default: 1000000)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='BS',
                        help='batch size (default: 10)')
    parser.add_argument('--num-sample', type=int, default=5, metavar='NS',
                        help='num sample (default: 5)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--train-file', type=str, default='/data1/nikhil/sentence-corpus/generator/train.txt', metavar='NS',
                        help='train file path')
    parser.add_argument('--save-at', type=str, default='/data1/nikhil/sentence-corpus/generator/', metavar='SA',
                        help='path where embedding files need to be saved')

    args = parser.parse_args()

    print 'Train File: %s' % args.train_file
    print 'Save Path: %s' % args.save_at

    # Flag changes the preprocessing steps for pkl files
    sentence_array = False

    if args.train_file.endswith('.txt'):
        data_files = [args.train_file]
        data = [open(file, "r").read() for file in data_files]
    
    elif args.train_file.endswith('.pkl'):
        data_files = args.train_file.split(',')
        print data_files
        data = [pkl.load(open(file, "rb")) for file in data_files]
        sentence_array = True
    
    idx_files = [args.save_at + '/words_vocab.pkl',
                      args.save_at + '/characters_vocab.pkl']

    batch_loader = BatchLoader(data, idx_files, train_embedding=True, sentence_array = sentence_array)
    
    params = Parameters(batch_loader.max_word_len,
                        batch_loader.max_seq_len,
                        batch_loader.words_vocab_size,
                        batch_loader.chars_vocab_size)

    neg_loss = NEG_loss(params.word_vocab_size, params.word_embed_size)
    if args.use_cuda:
        neg_loss = neg_loss.cuda()

    # NEG_loss is defined over two embedding matrixes with shape of [params.word_vocab_size, params.word_embed_size]
    optimizer = SGD(neg_loss.parameters(), 0.1)

    for iteration in range(args.num_iterations):

        input_idx, target_idx = batch_loader.next_embedding_seq(args.batch_size)

        input = Variable(t.from_numpy(input_idx).long())
        target = Variable(t.from_numpy(target_idx).long())
        if args.use_cuda:
            input, target = input.cuda(), target.cuda()

        out = neg_loss(input, target, args.num_sample).mean()

        optimizer.zero_grad()
        out.backward()
        optimizer.step()

        if iteration % 500 == 0:
            out = out.cpu().data.numpy()[0]
            print('iteration = {}, loss = {}'.format(iteration, out))

        if (1+iteration) % 100000 == 0: 
            word_embeddings = neg_loss.input_embeddings()
            np.save((args.save_at + '/word_embeddings_%d.npy'%iteration), word_embeddings)
    
