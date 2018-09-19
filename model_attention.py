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
        self.decoder = nn.Linear(nhid*2, ntoken)

        self.attn_matrix_left = nn.Linear(nhid, nhid)
        self.attn_linear_left = nn.Linear(nhid, 1)
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()
        self.attn_matrix_right = nn.Linear(nhid, nhid)
        self.attn_linear_right = nn.Linear(nhid, 1)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        #####################################################################################
        ################ Tying of weights not possible in bidirectional RNNs ################
        #####################################################################################
        # if tie_weights:
        #     if nhid != ninp:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight
        #####################################################################################
        #####################################################################################


        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
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

        output_left = self.drop(output_left)
        output_right = self.drop(output_right)

        outputs = torch.Tensor(output_left.size(0),output_left.size(1), self.ntoken)
        attn_weights = []

        for i in range(1, output_left.size(0)+1):

            attn_weights_left = self.drop(self.attn_matrix_left(
                output_left[:i].view(output_left[:i].size(0)*output_left[:i].size(1), output_left[:i].size(2))))
            attn_weights_left = self.attn_linear_left(self.tanh(attn_weights_left))
            attn_weights_left = self.softmax(attn_weights_left.view(output_left[:i].size(0),
                                                                    output_left[:i].size(1),
                                                                    attn_weights_left.size(1)))

            attn_weights_right = self.drop(self.attn_matrix_right(
                output_right[i-1:].view(output_right[i-1:].size(0) * output_right[i-1:].size(1),
                                        output_right[i-1:].size(2))))
            attn_weights_right = self.attn_linear_right(self.tanh(attn_weights_right))
            attn_weights_right = self.softmax(attn_weights_right.view(output_right[i-1:].size(0),
                                                                      output_right[i-1:].size(1),
                                                                      attn_weights_right.size(1)))

            # print(output_left[:i].size())
            # print(attn_weights_left.size())
            # print((output_left[:i]*attn_weights_left).size())
            context_left = torch.sum((output_left[:i]*attn_weights_left),0)
            # print(context_left.size())

            # print(output_right[i-1:].size())
            # print(attn_weights_right.size())
            # print((output_right[i-1:]*attn_weights_right).size())
            context_right = torch.sum((output_right[i-1:]*attn_weights_right), 0)
            # print(context_right.size())

            context = torch.cat((context_left, context_right), -1)
            decoded = self.decoder(context)

            outputs[i-1] = decoded
            attn_weights.append((attn_weights_left, attn_weights_right))
            #print(outputs)
            #input()

        return torch.Tensor(outputs), attn_weights


    def text_imputation(self, data_left, data_right, hidden_left, hidden_right):

        data_right = data_right.flip(0)

        emb_left = (self.encoder(data_left))
        emb_right = (self.encoder(data_right))

        output_left, hidden_left = self.rnn_left(emb_left, hidden_left)
        output_right, hidden_right = self.rnn_right(emb_right, hidden_right)

        output_right = output_right.flip(0)

        attn_weights_left = self.drop(self.attn_matrix_left(
            output_left.view(output_left.size(0) * output_left.size(1), output_left.size(2))))
        attn_weights_left = self.drop(self.attn_linear_left(self.tanh(attn_weights_left)))
        attn_weights_left = self.softmax(
            attn_weights_left.view(output_left.size(0), output_left.size(1), attn_weights_left.size(1)))

        attn_weights_right = self.drop(self.attn_matrix_right(
            output_right.view(output_right.size(0) * output_right.size(1),
                                      output_right.size(2))))
        attn_weights_right = self.drop(self.attn_linear_right(self.tanh(attn_weights_right)))
        attn_weights_right = self.softmax(
            attn_weights_right.view(output_right.size(0), output_right.size(1),
                                    attn_weights_right.size(1)))

        context_left = torch.sum((output_left * attn_weights_left), 0)
        context_right = torch.sum((output_right * attn_weights_right), 0)

        context = torch.cat((context_left, context_right), -1)
        decoded = self.decoder(context)
        # output_left = (output_left)[-1]
        # output_right = (output_right)[-1]
        # output = torch.cat((output_left, output_right), 1)

        # decoded = self.decoder(output)

        return decoded, attn_weights_left, attn_weights_right


    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
