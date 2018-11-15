import torch.nn as nn
import torch

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, sen_length, device, dropout=0.5, tie_weights=False):
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

        self.attn = nn.Linear(nhid*3,1)
        self.softmax = nn.Softmax(dim=0)

        self.transformer = nn.Linear(nhid*3, nhid)
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
        self.length = sen_length # Denotes the padding at the end of the sentence
        self.device = device
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

        output_left, hidden_left = self.rnn_left(emb_left, hidden_left)
        output_right, hidden_right = self.rnn_right(emb_right, hidden_right)

        output_right = output_right.flip(0)

        output = torch.zeros(output_left.size()).to(self.device)

        for i in range(0,output_left.size(0)):
            context = torch.cat((output_left[0:i+1], output_right[i:]),0)
            concat = self.drop(torch.cat((output_left[i].expand_as(context),
                                          output_right[i].expand_as(context), context), -1))
            attn_weight = self.tanh(self.attn(concat.view(concat.size(0)*concat.size(1), concat.size(2))))
            attn_weight = attn_weight.view(concat.size(0), concat.size(1), attn_weight.size(-1))
            attn_weight = self.softmax(attn_weight)
            context_vector = attn_weight*context
            context_vector = context_vector.sum(dim=0)
            concat2 = self.drop(torch.cat((output_left[i], output_right[i], context_vector), -1))
            predicted = self.tanh(self.transformer(concat2))
            output[i] = predicted

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))


    def text_imputation(self, data_left, data_right, hidden_left, hidden_right):

        data_right = data_right.flip(0)

        emb_left = self.encoder(data_left)
        emb_right = self.encoder(data_right)

        output_left, hidden_left = self.rnn_left(emb_left, hidden_left)
        output_right, hidden_right = self.rnn_right(emb_right, hidden_right)

        output_right = output_right.flip(0)

        context = torch.cat((output_left, output_right), 0)
        concat = torch.cat((output_left[-1].expand_as(context),
                                      output_right[0].expand_as(context), context), -1)
        attn_weight = self.tanh(self.attn(concat.view(concat.size(0) * concat.size(1), concat.size(2))))
        attn_weight = attn_weight.view(concat.size(0), concat.size(1), attn_weight.size(-1))
        attn_weight = self.softmax(attn_weight)
        context_vector = attn_weight * context
        context_vector = context_vector.sum(dim=0)
        concat2 = torch.cat((output_left[-1], output_right[0], context_vector), -1)
        predicted = self.tanh(self.transformer(concat2))
        decoded = self.decoder(predicted)

        return decoded, attn_weight


    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
