# -*- coding: utf-8 -*-

import collections
import os, sys
import re
import cPickle as pkl

import numpy as np

from batch_loader import BatchLoader
from .functional import *

"""
This code expects word_embeddings, word vocab, and char vocab already generated using 
train_embedding.py
"""

class DataHandler:

    def __init__(self, vocab_files, \
                 generator_file, \
                 discriminator_file, batch_size=32, load_generator=True):

        self.batch_size = batch_size

        with open(generator_file, 'rb') as f:
            self.generator_data = [pkl.load(f)]

        """
        with open(discriminator_file, 'rb') as f:
            self.discriminator_X, self.discriminator_Y = pkl.load(f)
        """
        self.gen_batch_loader = BatchLoader(self.generator_data, vocab_files, sentence_array=True)
        
        
