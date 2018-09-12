import argparse
from operator import itemgetter
import torch
import os
from nltk.corpus import stopwords
import context_data

stopWords = set(list(stopwords.words('english'))+['<eos>','<sos>'])
parser = argparse.ArgumentParser(description='PyTorch Context-filling Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.checkpoint, 'rb') as f:
    if args.cuda == False:
        model = torch.load(f,map_location='cpu')
    else:
        model = torch.load(f).to(device)
model.eval()

with open(args.model_left, 'rb') as f:
    model_left = torch.load(f).to(device)
model_left.eval()

with open(args.model_right, 'rb') as f:
    model_right = torch.load(f).to(device)
model_right.eval()

softmax = torch.nn.Softmax()

corpus = context_data.Corpus(args.data)
ntokens = len(corpus.dictionary)


hidden_left = model_left.init_hidden(1)
hidden_right = model_right.init_hidden(1)