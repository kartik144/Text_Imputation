import argparse
import torch
import pickle
from utils import msr_util
from utils import data_test
from utils.data_train import Dictionary
from utils.data_train_attn import Dictionary
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch Sentence Completion Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/msr',
                    help='location of the data corpus')
parser.add_argument('--model_bi', type=str, default='./models/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--model_attn', type=str, default='./models/model_attn.pt',
                    help='model checkpoint to use')
parser.add_argument('--dict', type=str, default='./Dictionary/dict_msr.pt',
                    help='path to pickled dictionary')
parser.add_argument('--dict_attn', type=str, default='./Dictionary/dict_msr_attn.pt',
                    help='path to pickled dictionary')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--file', type=str, default='data/msr_test/test.txt',
                    help='location of test data file')
parser.add_argument('--ans', type=str, default='data/msr_test/test_ans.txt',
                    help='location of answer file')
parser.add_argument('--sen_length', type=int,
                    default=50,
                    help='Threshold for limiting sentences of the data '
                         '(to restrict unnecessary long sentences)')
parser.add_argument('--case', action='store_true',
                        help='use to convert all words to lowercase')

args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

with open(args.model_bi, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

with open(args.model_attn, 'rb') as f:
    model_attn = torch.load(f, map_location=device)
model_attn.eval()

dictionary, threshold = pickle.load(open(args.dict, "rb"))
dict_attn, threshold_attn = pickle.load(open(args.dict_attn, "rb"))

ntokens = len(dictionary)
ntokens_attn = len(dict_attn)

data = msr_util.get_data(args.file, args.ans)
counter = msr_util.AccuracyCounter()
counter_attn = msr_util.AccuracyCounter()

dev_data = data[:int(len(data)/2)]
test_data = data[int(len(data)/2):]

for s in dev_data:
    try:
        sentence = s['sentence']
        options = s['options']

        if args.case:
            sentence = sentence.lower()
            for index in range(0,len(options)):
                options[index] = options[index].lower()

        left_ids, right_ids = data_test.tokenize_input(sentence, dictionary)
        hidden_left = model.init_hidden(1)
        hidden_right = model.init_hidden(1)
        input_left = torch.LongTensor(left_ids).view(-1, 1).to(device)
        input_right = torch.LongTensor(right_ids).view(-1, 1).to(device)
        outputs = model.text_imputation(input_left, input_right, hidden_left, hidden_right)
        output_flat = outputs.view(-1, ntokens)[-1]

        l, r = data_test.tokenize_input(sentence, dict_attn, args.sen_length)
        hidden_left = model_attn.init_hidden(1)
        hidden_right = model_attn.init_hidden(1)
        input_left = torch.LongTensor(l).view(-1, 1)
        input_right = torch.LongTensor(r).view(-1, 1)
        output, attn_weights = model_attn.text_imputation(input_left, input_right, hidden_left, hidden_right)
        output_flat_attn = output.view(-1, ntokens_attn)[-1]

        scores = msr_util.get_scores(output_flat, options, dictionary)
        scores_attn = msr_util.get_scores(output_flat_attn, options, dict_attn)

        if scores[0][0] == s['answer']:
            counter.correct_()
        else:
            counter.incorrect()

        if scores_attn[0][0] == s['answer']:
            counter_attn.correct_()
        else:
            counter_attn.incorrect()
            print(scores_attn)
            print(s['answer'])
            fig, ax = plt.subplots()
            sentence = sentence.replace("___", "")
            im = ax.matshow(
                attn_weights.view(attn_weights.size(0), -1)[:len(sentence.split()) + 2].t().detach().numpy())

            ax.set_xticks(np.arange(len(sentence.split()) + 2))
            ax.set_xticklabels([x for x in ["<sos>"] + sentence.split() + ["eos"]])

            fig.colorbar(im)
            plt.xticks(rotation="45")
            plt.show()
    except Exception as e:
        print(e)

counter.display_results()
counter_attn.display_results()