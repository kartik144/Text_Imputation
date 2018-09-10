import argparse
from operator import itemgetter
import torch
import os

import context_data


class Counter():
    def __init__(self):
        self.em = 0
        self.topV = 0
        self.topX = 0

    def update(self, missing_word):
        predicted_idx = [w[0] for w in missing_word]
        if predicted_idx[0] == corpus.test_target[index]:
            self.em += 1

        if corpus.test_target[index] in predicted_idx[:5]:
            self.topV += 1

        if corpus.test_target[index] in predicted_idx[:10]:
            self.topX += 1



parser = argparse.ArgumentParser(description='PyTorch Context-filling Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
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

def get_missing_word(input):
    missing_word = []
    for i in range(0, input.size()[-1]):
        # print(output_flat[i].data, end=", ")
        if len(missing_word) < 10:
            missing_word.append((i, input[i].data))
            missing_word.sort(key=itemgetter(1))
        else:
            if input[i].data > missing_word[0][1]:
                missing_word[0] = (i, input[i].data)
                missing_word.sort(key=itemgetter(1))

    return missing_word

def print_sentence_test(corpus, index):
    for w in corpus.test_left[index]:
        print(corpus.dictionary.idx2word[w], end=" ")
    print("___", end=" ")
    for w in corpus.test_right[index]:
        print(corpus.dictionary.idx2word[w], end=" ")

    print("\nTarget Word: {0}".format(corpus.dictionary.idx2word[corpus.test_target[index]]))

def print_predictions(corpus, missing_word):
    missing_word.reverse()  # Reverse list to arrange in descending order of scores

    for idx, _ in missing_word:
        print(corpus.dictionary.idx2word[idx], end=", ")
    print()

def print_results(label,counter, corpus):
    print(label)
    print("Exact match: {0}/{1} ({2:.2f}%)".format(counter.em, len(corpus.test_target), 100*(counter.em/len(corpus.test_target))))
    print("Target word in top 5 predicted words: {0}/{1} ({2:.2f}%)".format(counter.topV, len(corpus.test_target), 100*(counter.topV / len(corpus.test_target))))
    print("Target word in top 10 predicted words: {0}/{1} ({2:.2f}%)".format(counter.topX, len(corpus.test_target), 100*(counter.topX / len(corpus.test_target))))

with torch.no_grad():
    print("=" * 89)
    print("============================= Predicting words for test set =============================")
    print("=" * 89)

    bi_counter = Counter()
    left_counter = Counter()
    right_counter = Counter()

    for index, line in enumerate(corpus.test_right):

        input_left = torch.LongTensor(corpus.test_left[index]).view(-1,1).to(device)
        input_right = torch.LongTensor(line).view(-1,1).flip(0).to(device)

        outputs_left, hidden_left = model_left(input_left, hidden_left)
        outputs_right, hidden_right = model_left(input_right, hidden_right)

        output_flat_left = softmax(outputs_left.view(-1, ntokens)[-1])
        output_flat_right = softmax(outputs_right.view(-1, ntokens)[-1])
        output_flat = output_flat_left + output_flat_right

        missing_word = get_missing_word(output_flat)
        missing_word_left = get_missing_word(output_flat_left)
        missing_word_right = get_missing_word(output_flat_right)

        print_sentence_test(corpus, index)

        print("Candidate words (bidirectional):\t\t", end=" ")
        print_predictions(corpus, missing_word)
        bi_counter.update(missing_word)

        print("Candidate words (unidirectional-left):\t", end=" ")
        print_predictions(corpus, missing_word_left)
        left_counter.update(missing_word_left)

        print("Candidate words (unidirectional-right):\t", end=" ")
        print_predictions(corpus, missing_word_right)
        right_counter.update(missing_word_right)

        print()

    print("=" * 80)
    print("=============================== ACCURACY RESULTS ===============================")
    print("=" * 80)
    print_results("BIDIRECTIONAL", bi_counter, corpus)
    print("=" * 80)

    print("=" * 80)
    print_results("UNIDIRECTIONAL - LEFT", left_counter, corpus)
    print("=" * 80)

    print("=" * 80)
    print_results("UNIIDIRECTIONAL - RIGHT", right_counter, corpus)
    print("=" * 80)
    print("\n\n\n")

with open(os.path.join(args.data, "context-fill.txt"), "r") as f:
    print("=" * 89)
    print("========================= Predicting words for random sentences =========================")
    print("=" * 89)
    for index, line in enumerate(corpus.context_right):
        missing_word=[]
        input_left = torch.LongTensor(corpus.context_left[index]).view(-1, 1).to(device)
        input_right = torch.LongTensor(line).view(-1, 1).flip(0).to(device)

        outputs_left, hidden_left = model_left(input_left, hidden_left)
        outputs_right, hidden_right = model_left(input_right, hidden_right)

        output_flat_left = softmax(outputs_left.view(-1, ntokens)[-1])
        output_flat_right = softmax(outputs_right.view(-1, ntokens)[-1])
        output_flat = output_flat_left + output_flat_right

        missing_word = get_missing_word(output_flat)
        missing_word_left = get_missing_word(output_flat_left)
        missing_word_right = get_missing_word(output_flat_right)

        print(f.readline(),end="")

        print("Candidate words (bidirectional):\t\t", end=" ")
        print_predictions(corpus, missing_word)

        print("Candidate words (unidirectional-left):\t", end=" ")
        print_predictions(corpus, missing_word_left)

        print("Candidate words (unidirectional-right):\t", end=" ")
        print_predictions(corpus, missing_word_right)

        print()
