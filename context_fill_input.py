import argparse
from operator import itemgetter
import torch
import os
from nltk.corpus import stopwords
import context_data

def get_inputs(sentence):
    words = ['<sos>'] + sentence.split() + ['<eos>']
    for index, w in enumerate(words):
        if w not in corpus.dictionary.word2idx.keys():
            words[index] = '<unk>'
            print(w)

    ids_left = []
    ids_right = []
    flag = False
    for word in words:
        if word == "___":
            flag = True
            continue
        if flag == False:
            ids_left.append(corpus.dictionary.word2idx[word])
        else:
            ids_right.append(corpus.dictionary.word2idx[word])

    return ids_left, ids_right


def get_missing_word(input):
    missing_word = []
    for i in range(0, input.size()[-1]):
        if (corpus.dictionary.idx2word[i].lower() in stopWords) or ('.' in corpus.dictionary.idx2word[i]):
            continue
        elif len(missing_word) < 10:
            missing_word.append((i, input[i].data))
            missing_word.sort(key=itemgetter(1))
        else:
            if input[i].data > missing_word[0][1]:
                missing_word[0] = (i, input[i].data)
                missing_word.sort(key=itemgetter(1))

    return missing_word


def print_predictions(corpus, missing_word):
    missing_word.reverse()  # Reverse list to arrange in descending order of scores

    for idx, _ in missing_word:
        print(corpus.dictionary.idx2word[idx], end=", ")
    print()

stopWords = set(list(stopwords.words('english'))+['<eos>','<sos>', ',', ':',"\"", "?", "!","I", "A", "OK", "_", "mr","--", "-", ")", "\'", "("])

parser = argparse.ArgumentParser(description='PyTorch Context-filling Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--file', type=str, default='./context-fill-2.txt',
                    help='location of the file to fill in ')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--model_left', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--model_right', type=str, default='./model.pt',
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
        model = torch.load(f, map_location = device)
model.eval()

with open(args.model_left, 'rb') as f:
    model_left = torch.load(f, map_location = device)
model_left.eval()

with open(args.model_right, 'rb') as f:
    model_right = torch.load(f, map_location = device)
model_right.eval()

softmax = torch.nn.Softmax(dim=0)

corpus = context_data.Corpus(args.data)
ntokens = len(corpus.dictionary)

sentence = input("Enter sentence (Enter $TOP to stop)\n")
while(sentence != "$TOP"):

    left_ids, right_ids = get_inputs(sentence)

    hidden_left = model_left.init_hidden(1)
    hidden_right = model_right.init_hidden(1)
    input_left = torch.LongTensor(left_ids).view(-1, 1).to(device)
    input_right = torch.LongTensor(right_ids).view(-1, 1).flip(0).to(device)

    outputs_left, hidden_left = model_left(input_left, hidden_left)
    outputs_right, hidden_right = model_right(input_right, hidden_right)

    output_flat_left = softmax(outputs_left.view(-1, ntokens)[-1])
    output_flat_right = softmax(outputs_right.view(-1, ntokens)[-1])

    missing_word_left = get_missing_word(output_flat_left)
    missing_word_right = get_missing_word(output_flat_right)

    print("Candidate words (unidirectional-left):\t", end=" ")
    print_predictions(corpus, missing_word_left)

    print("Candidate words (unidirectional-right):\t", end=" ")
    print_predictions(corpus, missing_word_right)

    hidden_left = model.init_hidden(1)
    hidden_right = model.init_hidden(1)
    input_left = torch.LongTensor(left_ids).view(-1, 1).to(device)
    input_right = torch.LongTensor(right_ids).view(-1, 1).to(device)

    outputs = model.text_imputation(input_left, input_right, hidden_left, hidden_right)
    output_flat = softmax(outputs.view(-1, ntokens)[-1])

    missing_word = get_missing_word(output_flat)

    print("Candidate words (joint-model): \t\t", end="")
    print_predictions(corpus, missing_word)

    print()
    sentence = input("Enter sentence (Enter $TOP to stop)\n")