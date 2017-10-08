"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

import torch


class Beam(object):
    """Ordered beam of candidate outputs."""

    """ Input is beam_size, batch_loader object, cuda_flag """
    def __init__(self, size, batch_loader, cuda=False):
        
        self.size = size
        self.done = False
        self.pad = batch_loader.word_to_idx[batch_loader.pad_token]
        self.bos = batch_loader.word_to_idx[batch_loader.go_token]
        self.eos = batch_loader.word_to_idx[batch_loader.end_token]
                
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos


        # The softmax outputs
        # To be used in passing gradients back
        self.output_probs = []
        
        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # use this when returning the hypothesis 
        self.output_probs.append(workd_lk)
        
        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk * self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        # self.prevKs appends the (index of self.size best beams) out of the (last self.size beams)
        
        self.nextYs.append(bestScoresId - prev_k * num_words)
        # self.nextYs appends the (absolute index of top self.size words after this advance)
        
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print len(self.prevKs), len(self.nextYs), len(self.output_probs)
                
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
#         print "inside:", hyp
        
        return hyp[::-1]

    """
    Walk back to construct the full hypothesis

    Returns:
        * best hypothesis softmax output_probs 
    """
    def get_hyp_probs(self):

        hyp = []
        k = 0
        
        for j in range(len(self.prevKs) - 1, -1, -1):
            # Iterate through time steps
            # Best beam at time step j
            k = self.prevKs[j][k]
            hyp.append(self.output_probs[j][k])
            # k = self.prevKs[j][k]

        return hyp[::-1]
        
