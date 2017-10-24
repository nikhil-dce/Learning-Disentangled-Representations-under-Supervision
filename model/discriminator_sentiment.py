__author__ = "Nikhil Mehta"
__copyright__ = "--"

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys

class Sentiment_CNN(nn.Module):
    
    def __init__(self, config):
        super(Sentiment_CNN, self).__init__()
        self.config = config

        # D = config.word_vocab_size
        D = config.word_embed_size
        print 'Embedding Dimension: ' + str(D)
        
        # Number of classes Positive/Negative
        C = 2
        print 'Number of classes: ' + str(C)

        # Channel Input
        Ci = 1

        # Number of kernels
        # Channel out
        Co = config.sentiment_kernel_num
        print 'Number of Kernels: ' + str(Co)
                
        # Kernel sizes
        Ks = config.sentiment_kernel_size
        print 'Kernel size: ' + str(Ks)

        # Embedding Object
        # self.embed = nn.Embedding(V, D)
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
                        
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''

        self.dropout = nn.Dropout(config.sentiment_dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
        
    def forward(self, x):

        """
        x should be (batch_size, seq_len, D)
        """

        x = x.unsqueeze(1) # (N,Ci,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit
