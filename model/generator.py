"""
Code Authored By:
Nikhil Mehta
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, params):
        super(Generator, self).__init__()

        self.params = params

        self.rnn = nn.LSTM(input_size = (self.params.latent_variable_size+1) + self.params.word_embed_size,
                           hidden_size = self.params.decoder_rnn_size,
                           num_layers = self.params.decoder_num_layers,
                           batch_first = True)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

    def only_decoder_beam (self, generator_input, zc, drop_prob, initial_state = None):

        [beam_batch_size, _, _] = generator_input.size()
        
        # generator_input = F.dropout(generator_input, drop_prob)
        zc = zc.unsqueeze(1)
        zc = t.cat([zc] * (beam_batch_size/zc.size(0)), 0)
        zc = zc.contiguous().view(beam_batch_size, 1, -1)
        
        # zc.size() => (beam_batch_size X 1 X latent_variable_size+c_size)
                
        # zc = zc.unsqueeze(0)
        # zc = t.cat([zc] * beam_batch_size, 0)
        
        generator_input = t.cat([generator_input, zc], 2)

        rnn_out, final_state = self.rnn(generator_input, initial_state)

        return rnn_out, final_state

    def forward(self, generator_input, zc, drop_prob, initial_state=None):

        #assert parameters_allocation_check(self), \
        #    'Parameter Check Fail'

        [batch_size, seq_len, _] = generator_input.size()

        generator_input = F.dropout(generator_input, drop_prob)

        zc = t.cat([zc] * seq_len, 1).view(batch_size, seq_len,  self.params.latent_variable_size+1)
                
        generator_input = t.cat((generator_input, zc), 2)
          
        rnn_out, final_state = self.rnn(generator_input, initial_state)
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state
