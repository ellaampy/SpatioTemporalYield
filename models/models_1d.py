from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, Sequential, ReLU


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MLP(nn.Module):
    def __init__(self, input_dim=15, sequence_len=46, hidden_dim1=512, hidden_dim2 =256, dropout = 0.2):
        super(MLP, self).__init__()
        self.modelname = f"FCN_input-dim={input_dim}_sequence_len={sequence_len}_hidden-dim1={hidden_dim1}_" \
                    f"hidden-dims2={hidden_dim2}_dropout={dropout}"

        self.input_dim = input_dim
        self.sequence_len = sequence_len
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.input_dim*self.sequence_len, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x =  x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x




    
class LSTM(torch.nn.Module):
    def __init__(self, input_dim=15, num_classes=1, hidden_dims=128, num_layers=4, dropout=0.5713020228087161, bidirectional=True, use_layernorm=True):
        self.modelname = f"LSTM_input-dim={input_dim}_num-classes={num_classes}_hidden-dims={hidden_dims}_" \
                         f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" \
                         f"_dropout={dropout}"

        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.d_model = num_layers * hidden_dims

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) * num_layers)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            hidden_dims = hidden_dims * 2

        self.linear_class = nn.Linear(hidden_dims * num_layers, num_classes, bias=True)


    def logits(self, x):

        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs, last_state_list = self.lstm.forward(x)

        h, c = last_state_list

        nlayers, batchsize, n_hidden = c.shape
        h = self.clayernorm(c.transpose(0, 1).contiguous().view(batchsize, nlayers * n_hidden))
        logits = self.linear_class.forward(h)

        return logits

    def forward(self, x):
        x = self.logits(x)
        x =  x.squeeze(-1)
        return x


# source attention: https://colab.research.google.com/github/ngduyanhece/nlp-tutorial/blob/master/4-3.Bi-LSTM%28Attention%29/Bi_LSTM%28Attention%29_Torch.ipynb    
class LSTM_Attn(torch.nn.Module):
    def __init__(self, input_dim=15, num_classes=1, hidden_dims=128, num_layers=6, dropout=0.5713020228087161, bidirectional=True, use_layernorm=True):
        self.modelname = f"LSTMAttn_input-dim={input_dim}_num-classes={num_classes}_hidden-dims={hidden_dims}_" \
                         f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" \
                         f"_dropout={dropout}"

        super(LSTM_Attn, self).__init__()

        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.d_model = num_layers * hidden_dims
        self.num_layers = num_layers

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) * num_layers)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            self.hidden_dims = self.hidden_dims * 2

        self.linear_class = nn.Linear(self.hidden_dims * num_layers, num_classes, bias=True)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        # hidden = final_state.view(-1, self.hidden_dims * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, final_state).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)
        return context, soft_attn_weights.data.cpu() # context : [batch_size, n_hidden * num_directions(=2)]
    

    def logits(self, x):

        if self.use_layernorm:
            x = self.inlayernorm(x)
        # print('initial size', x.shape)

        outputs, last_state_list = self.lstm.forward(x)
        # print('lstm output',outputs.shape)

        h, c = last_state_list
        # print('hidden state', h.shape, 'cell state', c.shape)

        nlayers, batchsize, n_hidden = c.shape 
        
        h = h.view(batchsize, self.hidden_dims, -1)
        # print('hidden state before attention', h.shape)
        
        #attention block
        context, attention = self.attention_net(outputs, h)
        # print('attn_output', context.shape, 'attention',attention.shape)
        
        context = self.clayernorm(context.contiguous().view(batchsize, nlayers * n_hidden))
        # print('final state', context.shape)
        
        logits = self.linear_class.forward(context)

        return logits
    
    
    def forward(self, x):
        x = self.logits(x)
        x =  x.squeeze(-1)
        # print(x.shape)
        return x
    


class LSTM_Attn_mod(nn.Module):
    def __init__(self, input_dim=15, num_classes=1, hidden_dims=128, num_layers=6, dropout=0.5713020228087161, bidirectional=True, use_layernorm=True):
        super(LSTM_Attn_mod, self).__init__()

        self.modelname = f"LSTMAttnMod_input-dim={input_dim}_num-classes={num_classes}_hidden-dims={hidden_dims}_" \
                         f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" \
                         f"_dropout={dropout}"
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.num_layers = num_layers

        if bidirectional:
            self.hidden_dims *= 2  # Adjust hidden_dims for bidirectional

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm(self.hidden_dims)  # Adjusted to match hidden_dims

        self.linear_class = nn.Linear(self.hidden_dims, num_classes)

    def attention_net(self, lstm_output, final_state):
        # Adjust final_state shape for matrix multiplication
        final_state = final_state.permute(1, 0, 2)  # [batch_size, num_layers * hidden_dim, 1]
        final_state = final_state.contiguous().view(lstm_output.size(0), -1, 1)  # [batch_size, hidden_dim, 1]

        # Compute attention weights
        attn_weights = torch.bmm(lstm_output, final_state).squeeze(2)  # [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # Apply softmax along the sequence dimension

        # Compute context vector
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2))  # [batch_size, hidden_dim, 1]
        return context.squeeze(2), soft_attn_weights

    def logits(self, x):
        if self.use_layernorm:
            x = self.inlayernorm(x)
        
        outputs, (h, c) = self.lstm(x)
        
        # h is [num_layers * num_directions, batch_size, hidden_dim]
        h = h[-1]  # Take the last layer hidden state

        # Attention block
        context, attention = self.attention_net(outputs, h.unsqueeze(2))
        
        # Layer normalization and linear layer
        if self.use_layernorm:
            context = self.clayernorm(context)  # No need to reshape
        logits = self.linear_class(context)

        return logits

    def forward(self, x):
        logits = self.logits(x)
        return logits.squeeze(-1)




class TempCNN(torch.nn.Module):
    def __init__(self, input_dim=15, num_classes=1, sequencelength=45, kernel_size=7, hidden_dims=128, dropout=0.18203942949809093):
        super(TempCNN, self).__init__()
        self.modelname = f"TempCNN_input-dim={input_dim}_num-classes={num_classes}_sequencelenght={sequencelength}_" \
                         f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = Flatten()
        self.dense1 = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength, 4 * hidden_dims, drop_probability=dropout)
        self.dense2 = nn.Sequential(nn.Linear(4 * hidden_dims, num_classes))

    def forward(self, x):
        # require NxTxD
        x = x.transpose(1,2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x) 
        return x.squeeze(-1)


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )
    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    



# https://github.com/MarcCoru/BreizhCrops/blob/master/breizhcrops/models/TransformerModel.py
class TransformerModel(nn.Module):
    def __init__(self, input_dim=13, num_classes=9, d_model=64, n_head=2, n_layers=5,
                 d_inner=128, activation="relu", dropout=0.017998950510888446):

        super(TransformerModel, self).__init__()
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)

        self.inlinear = Linear(input_dim, d_model)
        self.relu = ReLU()
        self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)
        self.flatten = Flatten()
        self.outlinear = Linear(d_model, num_classes)

        """
        self.sequential = Sequential(
            ,
            ,
            ,
            ,
            ReLU(),

        )
        """

    def forward(self,x):
        x = self.inlinear(x)
        x = self.relu(x)
        x = x.transpose(0, 1) # N x T x D -> T x N x D
        x = self.transformerencoder(x)
        x = x.transpose(0, 1) # T x N x D -> N x T x D
        x = x.max(1)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        # logits = self.outlinear(x)
        # logprobabilities = F.log_softmax(logits, dim=-1)
        # return logprobabilities
        return x.squeeze(-1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


                                                           



class Hist1D(torch.nn.Module):
    def __init__(self, input_dim=15, seq_length=45, kernel_size=7, hidden_dims=128, num_layers=4, bidirectional=True, dropout=0.18203942949809093, bin_size =32, decoder_type='LSTM'):
        super(Hist1D, self).__init__()
        self.modelname = f"Hist1D_input-dim={input_dim}_sequencelenght={seq_length}_" \
                         f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}_numlayer={num_layers}_"\
                         f"birectional={bidirectional}_binsize_{bin_size}"

        self.input_dim = input_dim
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.seq_length = seq_length
        self.hidden_dims = hidden_dims
        self.decoder_type = decoder_type
        self.bin_size = bin_size
        self.tempcnn = TempCNN(self.input_dim*self.bin_size, 1,  self.seq_length, self.kernel_size, self.hidden_dims, self.dropout)
        self.lstm = LSTM(self.input_dim*self.bin_size, 1, self.hidden_dims, num_layers, self.dropout, bidirectional, use_layernorm=True)

    def forward(self, x):
        x = x.view(-1, self.input_dim*self.bin_size, self.seq_length) 
        # require batchxtimexfeatures
        x = x.transpose(1,2)
        
        if self.decoder_type == 'LSTM':
            x = self.lstm(x)
        elif self.decoder_type == 'TempCNN':
            x  = self.tempcnn(x)
        return x
