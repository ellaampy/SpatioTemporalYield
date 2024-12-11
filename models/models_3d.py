import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.models_1d import LSTM, LSTM_Attn, TempCNN, MLP



class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, *x.size()[2:])  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), *y.size()[1:])  # (org_axis1, org_axis2, output_size)

        return y
    
    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
class CNN_LSTM_3D(nn.Module):   
    def __init__(self, input_dim=10, sequence_len=46, dropout=0.5, num_layers=4, bidirectional=True, hidden_dim=256, decoder_type='TempCNN'):
        super(CNN_LSTM_3D, self).__init__()
        self.modelname = f"CNN_LSTM_3D_input-dim={input_dim}_sequence_len={sequence_len}_hidden-dim1={hidden_dim}_dropout={dropout}_decoder_type={decoder_type}_num_layers{num_layers}"
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.seq_length = sequence_len
        self.decoder_type = decoder_type
        self.dropout_flatten = nn.Dropout(dropout)
        

        self.cnn_layers = nn.Sequential(
            # Defining a 3D convolution layer
            TimeDistributed(nn.Conv2d(self.input_dim, 64, kernel_size=5, stride=1, padding=1)),
            TimeDistributed(nn.BatchNorm2d(64)),
            TimeDistributed(nn.ReLU(inplace=True)),
            TimeDistributed(nn.MaxPool2d(kernel_size=(2,2))),
            TimeDistributed(nn.Dropout2d(0.5)),
            
            # Defining another 3D convolution layer
            TimeDistributed(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)),
            TimeDistributed(nn.BatchNorm2d(128)),
            TimeDistributed(nn.ReLU(inplace=True)),
            TimeDistributed(nn.MaxPool2d(kernel_size=(2,2))),
            TimeDistributed(nn.Dropout2d(0.5)),
            
            # Defining another 3D convolution layer
            TimeDistributed(nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)),
            TimeDistributed(nn.BatchNorm2d(256)),
            TimeDistributed(nn.ReLU(inplace=True)),
            TimeDistributed(nn.MaxPool2d(kernel_size=(2,2))),
            TimeDistributed(nn.Dropout2d(0.5)),
            
        )
        self.flatten=TimeDistributed(Flatten())
        self.linear = TimeDistributed(nn.Linear(256*2*2, 256))  # for patch 64 use 256*8*8, for patch 32 use 256*4*4
        # self.linear = TimeDistributed(nn.Linear(128*2*2, 256))  # for patch 64 use 128*8*8, for patch 32 use 128*2*2 for kernel 5
        
        self.tempcnn = TempCNN(input_dim=256, num_classes=1, sequencelength=self.seq_length, kernel_size=5, \
                               hidden_dims=self.hidden_dim, dropout=self.dropout)
        self.lstm = LSTM(input_dim=256, num_classes=1, hidden_dims=self.hidden_dim, \
                         num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional, use_layernorm=True)
        self.lstm_attn = LSTM_Attn(input_dim=256, num_classes=1, hidden_dims=self.hidden_dim, num_layers=self.num_layers,\
                                   dropout=self.dropout, bidirectional=self.bidirectional, use_layernorm=True)


    # Defining the forward pass    
    def forward(self, x):
        # print('initial size', x.shape)
        x = self.cnn_layers(x)
        # print('after cnn', x.shape)
        x = self.flatten(x)
        # print('before linear', x.shape)
        x = self.linear(x)
        x = self.dropout_flatten(x)
        # print(x.shape)
        if self.decoder_type == 'TempCNN':
            x = self.tempcnn(x)
        elif self.decoder_type == 'LSTM':
            x = self.lstm(x)
        elif self.decoder_type == 'LSTM_Attn':
            x = self.lstm_attn(x)
            
        return x
    

    
class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, input_dim, dropout):
        super(C3D, self).__init__()

        self.modelname = 'C3D_dropout{}'.format(dropout)
        self.input_dim = input_dim

        self.conv1 = nn.Conv3d(self.input_dim, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        # self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, 1)
        self.fc6 = nn.Linear(2048, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, 1)

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        
        # requires a channel shift
        x = x.permute(0, 2, 1, 3, 4)
        # print('permuted', x.shape)
        

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)
        # print('after last pool', h.shape)

        # h = h.view(-1, 8192)
        h = h.view(-1, 2048)
        # print('after view')
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        # probs = self.softmax(logits)

        return logits.squeeze(dim=1)

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""

## code source: https://github.com/DavideA/c3d-pytorch/blob/master/C3D_model.py
