# -*- coding: utf-8 -*-

import collections
import os, sys
import re

import numpy as np
from six.moves import cPickle

from .functional import *


class BatchLoader:
    def __init__(self, data, idx_files, common_count = None, train_embedding=False, sentence_array = False):

        '''
            :properties

                data_files - array containing paths to data sources

                idx_files - array of paths to vocabulury files

                tensor_files - matrix with shape of [2, target_num] containing paths to files
                    with data represented as tensors
                    where first index in shape corresponds to types of representation of data,
                    i.e. word representation and character-aware representation

                blind_symbol - special symbol to fill spaces in every word in character-aware representation
                    to make all words be the same lenght
                pad_token - the same special symbol as blind_symbol, but in case of lines of words
                go_token - start of sequence symbol
                end_token - end of sequence symbol

                chars_vocab_size - number of unique characters
                idx_to_char - array of shape [chars_vocab_size] containing ordered list of inique characters
                char_to_idx - dictionary of shape [chars_vocab_size]
                    such that idx_to_char[char_to_idx[some_char]] = some_char
                    where some_char is such that idx_to_char contains it

                words_vocab_size, idx_to_word, word_to_idx - same as for characters

                max_word_len - maximum word length
                max_seq_len - maximum sequence length
                num_lines - num of lines in data with shape [target_num]

                word_tensor -  tensor of shape [target_num, num_lines, line_lenght] c
                    ontains word's indexes instead of words itself

                character_tensor - tensor of shape [target_num, num_lines, line_lenght, max_word_len].
                    Rows contain character indexes for every word in data

            :methods

                build_character_vocab(self, data) -> chars_vocab_size, idx_to_char, char_to_idx
                    chars_vocab_size - size of unique characters in corpus
                    idx_to_char - array of shape [chars_vocab_size] containing ordered list of inique characters
                    char_to_idx - dictionary of shape [chars_vocab_size]
                        such that idx_to_char[char_to_idx[some_char]] = some_char
                        where some_char is such that idx_to_char contains it

                build_word_vocab(self, sentences) -> words_vocab_size, idx_to_word, word_to_idx
                    same as for characters

                preprocess(self, data_files, idx_files, tensor_files) -> Void
                    preprocessed and initialized properties and then save them

                load_preprocessed(self, data_files, idx_files, tensor_files) -> Void
                    load and and initialized properties

                next_batch(self, batch_size, target_str) -> encoder_word_input, encoder_character_input, input_seq_len,
                        decoder_input, decoder_output
                    randomly sampled batch_size num of sequences for target from target_str.
                    fills sequences with pad tokens to made them the same lenght.
                    encoder_word_input and encoder_character_input have reversed order of the words
                        in case of performance
        '''

#         self.data_files = data_files
#         self.idx_files = idx_files

        self.blind_symbol = ''
        self.pad_token = '_'
        self.go_token = '>'
        self.end_token = '|'
        self.a_token = '?'

                
        idx_exists = fold(f_and,
                          [os.path.exists(file) for file in idx_files],
                          True)

        data_words=[]
        
        
        if train_embedding:
            print "BatchLoader Train Embedding...==========================>"
            self.preprocess(data, idx_files, is_sentence=sentence_array)
        elif idx_exists:
            print "BatchLoader Idx Exists...==========================>"
            self.load_preprocessed(data, idx_files, is_sentence=sentence_array)
            print('preprocessed data was found and loaded')
        else:
            print 'Error: dictionaries do not exist.'
            sys.exit()

        print len(self.word_to_idx)
        print len(self.char_to_idx) 
        
        self.word_embedding_index = 0
            
    def clean_whole_data(self, string):
        string = re.sub('^[\d\:]+ ', '', string, 0, re.M)
        string = re.sub('\n\s{11}', ' ', string, 0, re.M)
        string = re.sub('\n{2}', '\n', string, 0, re.M)

        return string.lower()

    def clean_str(self, string):
        '''
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        '''

        string = re.sub(r"[^가-힣A-Za-z0-9(),!?:;.\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r":", " : ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def build_character_vocab(self, data, is_sentence = False):
        
        if not is_sentence:
            # unique characters with blind symbol
            chars = list(set(data)) + [self.blind_symbol, self.pad_token, self.go_token, self.end_token]
            chars_vocab_size = len(chars)
            
        else:
            chars = set()
            for sen in data:
                chars = set(sen).union(chars)

            chars = list(chars) + [self.blind_symbol, self.pad_token, self.go_token, self.end_token]
            chars_vocab_size = len(chars)
            
        # mappings itself
        idx_to_char = chars
        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        return chars_vocab_size, idx_to_char, char_to_idx

    def build_word_vocab(self, sentences):

        # Build vocabulary
        word_counts = collections.Counter(sentences)

        # Mapping from index to word
        idx_to_word = [x[0] for x in word_counts.most_common()]
        idx_to_word = list(sorted(idx_to_word)) + [self.pad_token, self.go_token, self.end_token]

        words_vocab_size = len(idx_to_word)

        # Mapping from word to index
        word_to_idx = {x: i for i, x in enumerate(idx_to_word)}

        return words_vocab_size, idx_to_word, word_to_idx

    def preprocess(self, data, idx_files, is_sentence = False):
                
        if not is_sentence:
            merged_data = data[0] # using file 1 
            data_words = [[line.split() for line in target.split('\n')] for target in data]
            merged_data_words = merged_data.split()
        else:
            data_words = [[line.split() for line in target] for target in data]
            merged_data = data[0] # Use file 1 Not: this is not compatible with multiple training files as of now
            merged_data_words = []
            for line in merged_data:
                merged_data_words.extend(line.split())
                        
        self.chars_vocab_size, self.idx_to_char, self.char_to_idx = self.build_character_vocab(merged_data, is_sentence=is_sentence)

        with open(idx_files[1], 'wb') as f:
            cPickle.dump(self.idx_to_char, f)

        self.words_vocab_size, self.idx_to_word, self.word_to_idx = self.build_word_vocab(merged_data_words)
        
        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])
        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]

        with open(idx_files[0], 'wb') as f:
            cPickle.dump(self.idx_to_word, f)
            
        self.create_tensors(data_words)
        self.just_words = [word for line in self.word_tensor[0] for word in line]

        
    '''
    Requires data preprocessed data in the form of sentences 
    separated by '\n'
    '''
    def load_preprocessed(self, data, idx_files, is_sentence = False):

        # Following Data from training txt file
        # Data words for each line
        if not is_sentence:
            data_words = [[line.split() for line in target.split('\n')] for target in data]
        else: # Sentences already in array. No need for split ('\n')
            data_words = [[line.split() for line in target] for target in data]
            # Max sequence length
        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        # Num lines in data array
        self.num_lines = [len(target) for target in data_words]

        # Following data from word and char vocab
        # Simple array of words and chars
        [self.idx_to_word, self.idx_to_char] = [cPickle.load(open(file, "rb")) for file in idx_files]
        # Word vocab size and Chars vocab size
        [self.words_vocab_size, self.chars_vocab_size] = [len(idx) for idx in [self.idx_to_word, self.idx_to_char]]
        # Reverse mapping
        [self.word_to_idx, self.char_to_idx] = [dict(zip(idx, range(len(idx)))) for idx in
                                                [self.idx_to_word, self.idx_to_char]]
        # Max word len
        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])
        self.create_tensors(data_words)
        
    def create_tensors(self, data_words):
        self.word_tensor = np.array(
            [[list(map(self.word_to_idx.get, line)) for line in target] for target in data_words])
                
        self.character_tensor = np.array(
            [[list(map(self.encode_characters, line)) for line in target] for target in data_words])
        
    def next_batch(self, batch_size, target_str, start_index):
        # target = 0 if target_str == 'train' else 1
        
        target = 0
        indexes = np.array(range(start_index, start_index+batch_size))
 
        encoder_word_input = [self.word_tensor[target][index] for index in indexes]
        encoder_character_input = [self.character_tensor[target][index] for index in indexes]
        input_seq_len = [len(line) for line in encoder_word_input]
        max_input_seq_len = np.amax(input_seq_len)

        encoded_words = [[idx for idx in line] for line in encoder_word_input]
        decoder_word_input = [[self.word_to_idx[self.go_token]] + line for line in encoder_word_input]
        decoder_character_input = [[self.encode_characters(self.go_token)] + line for line in encoder_character_input]
        decoder_output = [line + [self.word_to_idx[self.end_token]] for line in encoded_words]

        # sorry
        for i, line in enumerate(decoder_word_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_word_input[i] = line + [self.word_to_idx[self.pad_token]] * to_add

        for i, line in enumerate(decoder_character_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_character_input[i] = line + [self.encode_characters(self.pad_token)] * to_add

        for i, line in enumerate(decoder_output):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_output[i] = line + [self.word_to_idx[self.pad_token]] * to_add

        for i, line in enumerate(encoder_word_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            encoder_word_input[i] = [self.word_to_idx[self.pad_token]] * to_add + line[::-1]

        for i, line in enumerate(encoder_character_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            encoder_character_input[i] = [self.encode_characters(self.pad_token)] * to_add + line[::-1]

        return np.array(encoder_word_input), np.array(encoder_character_input), \
               np.array(decoder_word_input), np.array(decoder_character_input), np.array(decoder_output)

    def next_embedding_seq(self, seq_len):
        """
        :return:
            tuple of input and output for word embedding learning,
            where input = [b, b, c, c, d, d, e, e]
            and output  = [a, c, b, d, d, e, d, g]
            for line [a, b, c, d, e, g] at index i
        """

        words_len = len(self.just_words)
        seq = [self.just_words[i % words_len]
               for i in np.arange(self.word_embedding_index, self.word_embedding_index + seq_len)]

        result = []
        for i in range(seq_len - 2):
            result.append([seq[i + 1], seq[i]])
            result.append([seq[i + 1], seq[i + 2]])

        self.word_embedding_index = (self.word_embedding_index + seq_len) % words_len - 2
        result = np.array(result)

        return result[:, 0], result[:, 1]

    def go_input(self, batch_size):
        go_word_input = [[self.word_to_idx[self.go_token]] for _ in range(batch_size)]
        go_character_input = [[self.encode_characters(self.go_token)] for _ in range(batch_size)]

        return np.array(go_word_input), np.array(go_character_input)

    def encode_word(self, idx):
        result = np.zeros(self.words_vocab_size)
        result[idx] = 1
        return result

    def decode_word(self, word_idx):
        word = self.idx_to_word[word_idx]
        return word

    def sample_word_from_distribution(self, distribution):
        ix = np.random.choice(range(self.words_vocab_size), p=distribution.ravel())
        x = np.zeros((self.words_vocab_size, 1))
        x[ix] = 1
        return self.idx_to_word[np.argmax(x)]

    def encode_characters(self, characters):
        word_len = len(characters)
        to_add = self.max_word_len - word_len
        characters_idx = [self.char_to_idx[i] for i in characters] + to_add * [self.char_to_idx['']]
        return characters_idx

    def decode_characters(self, characters_idx):
        characters = [self.idx_to_char[i] for i in characters_idx]
        return ''.join(characters)
