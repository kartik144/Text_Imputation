import argparse
import torch
import pickle
from utils import data_test
from utils import process
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Sentence Completion Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model_bi', type=str, default='./models/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--model_left', type=str, default='./models/model_left.pt',
                    help='model checkpoint to use')
parser.add_argument('--model_right', type=str, default='./models/model_right.pt',
                    help='model checkpoint to use')
parser.add_argument('--model_attn', type=str, default='./models/model_attn.pt',
                    help='model checkpoint to use')
parser.add_argument('--dict', type=str, default='./Dictionary/dict.pt',
                    help='path to pickled dictionary')
parser.add_argument('--dict_attn', type=str, default='./Dictionary/dict_attn.pt',
                    help='path to pickled dictionary')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--file', type=str, default='#stdin#',
                    help='use when giving inputs through file instead of STDIN')
parser.add_argument('--N', type=int, default=10,
                    help='denotes number of words displayed (top N words predicted are displayed)')
parser.add_argument('--sen_length', type=int,
                    default=50,
                    help='Threshold for limiting sentences of the data '
                         '(to restrict unnecessary long sentences)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.model_bi, 'rb') as f:
        model = torch.load(f, map_location = device)
model.eval()

with open(args.model_attn, 'rb') as f:
    model_attn = torch.load(f, map_location=device)
model_attn.eval()

with open(args.model_left, 'rb') as f:
    model_left = torch.load(f, map_location = device)
model_left.eval()

with open(args.model_right, 'rb') as f:
    model_right = torch.load(f, map_location = device)
model_right.eval()

dictionary, threshold = pickle.load(open(args.dict, "rb"))
dict_attn, threshold_attn = pickle.load(open(args.dict_attn, "rb"))
ntokens = len(dictionary)


def complete_sentence(sentence, index):
    left_ids, right_ids = data_test.tokenize_input(sentence, dictionary)
    hidden_left = model_left.init_hidden(1)
    hidden_right = model_right.init_hidden(1)

    input_left = torch.LongTensor(left_ids).view(-1, 1).to(device)
    input_right = torch.LongTensor(right_ids).view(-1, 1).flip(0).to(device)

    outputs_left, hidden_left = model_left(input_left, hidden_left)
    outputs_right, hidden_right = model_right(input_right, hidden_right)

    output_flat_left = outputs_left.view(-1, ntokens)[-1]
    output_flat_right = outputs_right.view(-1, ntokens)[-1]
    output_flat = output_flat_left + output_flat_right

    missing_word = process.get_missing_word(output_flat, dictionary, args.N)
    missing_word_left = process.get_missing_word(output_flat_left, dictionary, args.N)
    missing_word_right = process.get_missing_word(output_flat_right, dictionary, args.N)

    # print("Candidate words (bidirectional):\t\t", end=" ")
    # process.print_predictions(dictionary, missing_word)

    print("Candidate words (unidirectional-left):\t", end=" ")
    process.print_predictions(dictionary, missing_word_left)

    print("Candidate words (unidirectional-right):\t", end=" ")
    process.print_predictions(dictionary, missing_word_right)

    hidden_left = model.init_hidden(1)
    hidden_right = model.init_hidden(1)
    input_left = torch.LongTensor(left_ids).view(-1, 1).to(device)
    input_right = torch.LongTensor(right_ids).view(-1, 1).to(device)

    outputs = model.text_imputation(input_left, input_right, hidden_left, hidden_right)
    output_flat = outputs.view(-1, ntokens)[-1]  # check this

    missing_word = process.get_missing_word(output_flat, dictionary, args.N)

    print("Candidate words (joint-model): \t\t", end="")
    process.print_predictions(dictionary, missing_word)

    ntokens_attn = len(dict_attn)
    l, r = data_test.tokenize_input(sentence, dict_attn, args.sen_length)
    hidden_left = model_attn.init_hidden(1)
    hidden_right = model_attn.init_hidden(1)
    input_left = torch.LongTensor(l).view(-1, 1)
    input_right = torch.LongTensor(r).view(-1, 1)
    output, attn_weights = model_attn.text_imputation(input_left, input_right, hidden_left, hidden_right)
    output_flat = output.view(-1, ntokens_attn)[-1]
    missing_word = process.get_missing_word(output_flat, dict_attn, args.N)
    print("Candidate words (attn): \t\t", end="")
    process.print_predictions(dict_attn, missing_word)

    fig, ax = plt.subplots()
    sentence = sentence.replace("___", "")
    im = ax.matshow(attn_weights.view(attn_weights.size(0), -1)[:len(sentence.split()) + 2].t().detach().numpy())

    ax.set_xticks(np.arange(len(sentence.split()) + 2))
    ax.set_xticklabels([x for x in ["<sos>"] + sentence.split() + ["eos"]])

    fig.colorbar(im)
    plt.xticks(rotation="45")

    if index != 0:
        plt.savefig('Attention_images/{0}.png'.format(index))
        plt.close()
    else:
        plt.show()

    print()


if args.file == '#stdin#':

    sentence = input("Enter sentence (Enter $TOP to stop)\n")
    while sentence != "$TOP":
        try:
            complete_sentence(sentence, 0)
        except Exception as e:
            print(e)

        sentence = input("Enter sentence (Enter $TOP to stop)\n")

else:

    with open(args.file, "r") as f:
        index = 0
        for line in f:
            index += 1
            print(str(index)+". "+line, end="")
            try:
                complete_sentence(line, index)
            except Exception as e:
                print(e)