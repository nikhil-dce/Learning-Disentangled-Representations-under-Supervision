import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .decoder import Decoder
from .encoder import Encoder

from selfModules.embedding import Embedding

from utils.functional import kld_coef, parameters_allocation_check, fold
from utils.beam_search import Beam

class RVAE(nn.Module):

    def __init__(self, params, embedding_path):
        super(RVAE, self).__init__()

        self.params = params
        
        self.embedding = Embedding(self.params, embedding_path) 
        
        self.encoder = Encoder(self.params)
        
        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        
        self.decoder = Decoder(self.params)         

    def forward(self, drop_prob,
                encoder_word_input=None, encoder_character_input=None,
                decoder_word_input=None, decoder_character_input=None,
                z=None, initial_state=None):

        """
        :param encoder_word_input: A tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: A tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: A tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is to be performed

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
                                  [encoder_word_input, encoder_character_input, decoder_word_input],
                                  True) \
            or (z is not None and decoder_word_input is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            
            # Get context from encoder and sample z ~ N(mu, std)
            
            [batch_size, _] = encoder_word_input.size()

            encoder_input = self.embedding(encoder_word_input, encoder_character_input)
                        
            context , h_0 , c_0 = self.encoder(encoder_input, None)
            State = (h_0, c_0) #Final state of Encoder
            
            # context_2 , _ , _ = self.encoder_2( encoder_input_2, State )   #Encoder_2 for Ques_2
            
            # mu = self.context_to_mu(context_2)
            # logvar = self.context_to_logvar(context_2)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()

            # encoder_input = self.embedding(encoder_word_input, encoder_character_input)
            # _ , h_0 , c_0 = self.encoder_3(encoder_input, None)
            # initial_state = (h_0,c_0) #Final state of Encoder-1

        else:
            kld = None
            mu = None
            std = None

        decoder_input = self.embedding.word_embed(decoder_word_input)   
        out, final_state = self.decoder(decoder_input, z, drop_prob, initial_state)        

        return out, final_state, kld, mu, std

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        def train(i, batch_size, use_cuda, dropout, start_index):
            input = batch_loader.next_batch(batch_size, 'train', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            logits, _, kld,_ ,_ = self(dropout,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input, decoder_character_input,
                                  z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            loss = 79 * cross_entropy + kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, kld, kld_coef(i)

        return train

    def validater(self, batch_loader):
        def validate(batch_size, use_cuda, start_index):
            input = batch_loader.next_batch(batch_size, 'valid', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            logits, _, kld,_ ,_ = self(0.,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input, decoder_character_input,
                                  z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target)

            return cross_entropy, kld

        return validate

    def sample(self, batch_loader, seq_len, seed, use_cuda, State):

        seed = Variable(seed)
        if use_cuda:
            seed = seed.cuda()

        print seed.size()
        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        result = ''

        initial_state = State

        for i in range(seq_len):
            #logits, initial_state, _ ,_,_= self(0., None, None,
            #                                decoder_word_input, decoder_character_input,
            #                                seed, initial_state)

            decoder_input = self.embedding.word_embed(decoder_word_input)   
            logits, initial_state = self.decoder(decoder_input, seed, 0., initial_state)        
            
            logits = logits.view(-1, self.params.word_vocab_size)
            prediction = F.softmax(logits)

            word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

            if word == batch_loader.end_token:
                break

            result += ' ' + word

            decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
            decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

            decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
            decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

            if use_cuda:
                decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        return result

    def sampler(self, batch_loader, seq_len, seed, use_cuda, i, beam_size, n_best):

        input = batch_loader.next_batch(1, 'test', i)
#         print input
        input = [Variable(t.from_numpy(var)) for var in input]
        input = [var.long() for var in input]
        input = [var.cuda() if use_cuda else var for var in input]
        [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

        encoder_input = self.embedding(encoder_word_input, encoder_character_input)

        _ , h0 , c0 = self.encoder_3(encoder_input, None)
        State = (h0,c0)

        results, scores = self.sample_beam(batch_loader, seq_len, seed, use_cuda, State, beam_size, n_best)

        State = None
                
        return results, scores


    def sample_beam_for_decoder (self, batch_loader, seq_len, seed,
                                 use_cuda, beam_size = 10, n_best = 5, samples=1):

        seed = Variable(seed)
        if use_cuda:
            seed = seed.cuda()

        # State see the shape

        dec_states = None
        drop_prob = 0.0
        batch_size = samples # Make this samples; fix the error
        
        beam = [Beam(beam_size, batch_loader, cuda=True) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        
        for i in range(seq_len):
            
            input = t.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)
                        
            trg_emb = self.embedding.word_embed(Variable(input).transpose(1, 0))

            trg_h, dec_states = self.decoder.only_decoder_beam(trg_emb, seed, drop_prob, dec_states)

            dec_out = trg_h.squeeze(1)
                        
            out = F.softmax(self.decoder.fc(dec_out)).unsqueeze(0)
                        
            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]
                
                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx] # Why access [:,:,idx]

                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = t.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.params.decoder_rnn_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            remaining_sents = len(active) 

         # (4) package everything up

        allHyp, allScores = [], []


        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        word_hyp = []
        result = []

        all_sent_codes = np.asarray(allHyp)
        all_sent_codes = np.transpose(allHyp, (0,2,1))

        all_sentences = []
        for batch in all_sent_codes:
            sentences = []
            for i_best in batch:
                sentence = ""
                for word_code in i_best:
                    word = batch_loader.decode_word(word_code)
                    if word == batch_loader.end_token:
                        break
                    sentence += ' ' + word
                sentences.append(sentence)

            all_sentences.extend(sentences)
        

        """
        for hyp in allHyp:

            sentence = []
            
            for i_step in range(seq_len):

                for idx in range(n_best):

                    if sentence[i_step][idx] == batch_loader.end_token:
                        continue;

                    
                for word_idx in hyp[i_step]:

                    if sentence[i_step][]
                    word = batch_loader.decode_word(word_idx)
                    # temp = map(batch_loader.decode_word, hyp[i_step])

                    if word == batch_loader.end_token:
                        break

                result += ' ' + word

                
                print temp
                word_hyp += temp
                
        """
            
        return all_sentences, allScores 

        
    
    def sample_beam(self, batch_loader, seq_len, seed, use_cuda, State, beam_size, n_best):

        if use_cuda:
            seed = seed.cuda()

        dec_states = State
 
        dec_states = [
            dec_states[0].repeat(1, beam_size, 1),
            dec_states[1].repeat(1, beam_size, 1)
        ]

        drop_prob = 0.0
        beam_size = beam_size
        batch_size = 1  
        
        beam = [Beam(beam_size, batch_loader, cuda=True) for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        
        for i in range(seq_len):
            
            input = t.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(1, -1)

            trg_emb = self.embedding.word_embed(Variable(input).transpose(1, 0))
 
            trg_h, dec_states = self.decoder.only_decoder_beam(trg_emb, seed, drop_prob, dec_states)
            dec_out = trg_h.squeeze(1)
            out = F.softmax(self.decoder.fc(dec_out)).unsqueeze(0)

            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = t.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.params.decoder_rnn_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            remaining_sents = len(active) 

         # (4) package everything up

        allHyp, allScores = [], []


        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        return allHyp, allScores 
