import argparse
from operator import itemgetter
import torch
import os
from nltk.corpus import stopwords
import context_data

def get_dataset(filename, corpus):
    with open(filename, "r") as f:
        context_left = []
        context_right = []

        for line in f:
            words = ['<sos>'] + line.split() + ['<eos>']
            for index, w in enumerate(words):
                if w not in corpus.dictionary.word2idx.keys():
                    words[index] = '<unk>'
                    # print(w)

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

            context_left.append(ids_left)
            context_right.append(ids_right)

        return context_left, context_right


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

    for idx, _ in missing_word:
        print(corpus.dictionary.idx2word[idx], end=", ")
    print()

stopWords = set(list(stopwords.words('english'))+['<eos>','<sos>', ',', ':',"\"", "?", "!","I", "A", "OK", "_", "mr"])

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

context_left, context_right = get_dataset("context-fill-2.txt", corpus)

criterion = torch.nn.CrossEntropyLoss()

with open(os.path.join(args.file), "r") as f:
    for index, line in enumerate(context_left):

        # hidden_right = model_right.init_hidden(1)
        words = []
        for w in range(0, len(corpus.dictionary.idx2word)):
            sentence = line + [w] + context_right[index]
            inputs = sentence[:-1]
            target = torch.LongTensor(sentence[1:]).view(-1).to(device)
            input_left = torch.LongTensor(inputs).view(-1, 1).to(device)
            hidden_left = model_left.init_hidden(1)
            outputs_left, hidden_left = model_left(input_left, hidden_left)
            loss = criterion(outputs_left.view(-1, ntokens), target)

            if len(words)<10:
                words.append((w,loss.item()))
                words.sort(key=itemgetter(1))
            else:
                if loss.item() < words[-1][1]:
                    words[-1] = (w,loss.item())
                    words.sort(key=itemgetter(1))

        print_predictions(corpus, words)

        # input_left = torch.LongTensor(context_left[index]).view(-1, 1).to(device)
        # input_right = torch.LongTensor(line).view(-1, 1).flip(0).to(device)
        #
        # outputs_left, hidden_left = model_left(input_left, hidden_left)
        # outputs_right, hidden_right = model_left(input_right, hidden_right)
        #
        # output_flat_left = softmax(outputs_left.view(-1, ntokens)[-1])
        # output_flat_right = softmax(outputs_right.view(-1, ntokens)[-1])
        # output_flat = output_flat_left + output_flat_right
        #
        # missing_word = get_missing_word(output_flat)
        # missing_word_left = get_missing_word(output_flat_left)
        # missing_word_right = get_missing_word(output_flat_right)
        #
        # print(f.readline(), end="")
        #
        # print("Candidate words (bidirectional):\t\t", end=" ")
        # print_predictions(corpus, missing_word)
        #
        # print("Candidate words (unidirectional-left):\t", end=" ")
        # print_predictions(corpus, missing_word_left)
        #
        # print("Candidate words (unidirectional-right):\t", end=" ")
        # print_predictions(corpus, missing_word_right)
        #
        # hidden_left = model.init_hidden(1)
        # hidden_right = model.init_hidden(1)
        # input_left = torch.LongTensor(corpus.context_left[index]).view(-1, 1).to(device)
        # input_right = torch.LongTensor(corpus.context_right[index]).view(-1, 1).to(device)
        #
        # outputs = model.text_imputation(input_left, input_right, hidden_left, hidden_right)
        # output_flat = softmax(outputs.view(-1, ntokens)[-1])
        #
        # missing_word = get_missing_word(output_flat)
        #
        # print("Candidate words (joint-model): \t\t", end="")
        # print_predictions(corpus, missing_word)

        # print()