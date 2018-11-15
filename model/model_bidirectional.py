import torch.nn as nn
import torch

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn_left = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn_right = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn_left = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            self.rnn_right = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.transformer = nn.Linear(nhid*2, nhid)
        self.tanh = nn.Tanh()
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        ##############################################################################################################
        ################ Tying of weights not possible in bidirectional RNNs in case of concatenation ################
        ##############################################################################################################
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        ##############################################################################################################
        ##############################################################################################################


        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.bidirectional = True

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, data_left, data_right, hidden_left, hidden_right):

        data_right = data_right.flip(0)

        emb_left = self.drop(self.encoder(data_left))
        emb_right = self.drop(self.encoder(data_right))

        output_left , hidden_left = self.rnn_left(emb_left, hidden_left)
        output_right, hidden_right = self.rnn_right(emb_right, hidden_right)

        output_right = output_right.flip(0)

        # output_left = self.drop(output_left)
        # output_right = self.drop(output_right)

        output = self.drop(torch.cat((output_left, output_right), -1))
        output = self.tanh(self.transformer(output))

        # output_left = self.drop(output_left)
        # output_right = self.drop(output_right)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        # decoded_left = self.decoder(output_left.view(output_left.size(0)*output_left.size(1), output_left.size(2)))
        # decoded_right = self.decoder(output_right.view(output_right.size(0)*output_right.size(1), output_right.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)) #, decoded_left.view(output_left.size(0), output_left.size(1), decoded_left.size(1)), decoded_right.view(output_right.size(0), output_right.size(1), decoded_right.size(1))


    def text_imputation(self, data_left, data_right, hidden_left, hidden_right):

        data_right = data_right.flip(0)

        emb_left = self.encoder(data_left)
        emb_right = self.encoder(data_right)

        output_left, hidden_left = self.rnn_left(emb_left, hidden_left)
        output_right, hidden_right = self.rnn_right(emb_right, hidden_right)

        output_right = output_right.flip(0)

        output_left = output_left[-1]
        output_right = output_right[0]
        output = torch.cat((output_left, output_right), -1)

        output = self.tanh(self.transformer(output))
        decoded = self.decoder(output)

        return decoded


    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
