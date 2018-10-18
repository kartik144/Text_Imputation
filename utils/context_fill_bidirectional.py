import argparse
from operator import itemgetter
import torch
import os
from nltk.corpus import stopwords
import data

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

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

def get_missing_word(input):
    missing_word = []
    for i in range(0, input.size()[-1]):
        if corpus.dictionary.idx2word[i] in stopWords:
            continue
        elif len(missing_word) < 10:
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

with torch.no_grad():
    print("=" * 89)
    print("============================= Predicting words for test set =============================")
    print("=" * 89)
    em=0
    topV=0
    topX=0
    for index in range(0, len(corpus.test_left)):
        hidden_left = model.init_hidden(1)
        hidden_right = model.init_hidden(1)
        input_left=torch.LongTensor(corpus.test_left[index]).view(-1,1).to(device)
        input_right = torch.LongTensor(corpus.test_right[index]).view(-1, 1).to(device)

        outputs = model.text_imputation(input_left, input_right, hidden_left, hidden_right)
        output_flat = outputs[-1]

        missing_word = get_missing_word(output_flat)
        print_sentence_test(corpus, index)

        print("Candidate words: ", end= " ")

        missing_word.reverse() # Reverse list to arrange in descending order of scores

        for idx, _ in missing_word:
            print(corpus.dictionary.idx2word[idx], end=", ")
        print("\n")

        predicted_idx = [w[0] for w in missing_word]
        if predicted_idx[0] == corpus.test_target[index]:
            em+=1

        if corpus.test_target[index] in predicted_idx[:5]:
            topV+=1

        if corpus.test_target[index] in predicted_idx[:10]:
            topX += 1

    print("=" * 80)
    print("=============================== ACCURACY RESULTS ===============================")
    print("=" * 80)
    print("Exact match: {0}/{1} ({2:.2f}%)".format(em, len(corpus.test_target), 100*(em/len(corpus.test_target))))
    print("Target word in top 5 predicted words: {0}/{1} ({2:.2f}%)".format(topV, len(corpus.test_target), 100*(topV / len(corpus.test_target))))
    print("Target word in top 10 predicted words: {0}/{1} ({2:.2f}%)".format(topX, len(corpus.test_target), 100*(topX / len(corpus.test_target))))
    print("=" * 80)
    print("\n\n\n")

with open(os.path.join(args.data, "context_fill"), "r") as f:
    print("=" * 89)
    print("========================= Predicting words for random sentences =========================")
    print("=" * 89)
    for index in range(0, len(corpus.context_left)):
        hidden_left = model.init_hidden(1)
        hidden_right = model.init_hidden(1)
        input_left = torch.LongTensor(corpus.context_left[index]).view(-1,1).to(device)
        input_right = torch.LongTensor(corpus.context_right[index]).view(-1, 1).to(device)

        print(f.readline(), end="")
        outputs = model.text_imputation(input_left, input_right, hidden_left, hidden_right)
        output_flat = outputs[-1]

        missing_word = get_missing_word(output_flat)

        print("Candidate words: ",end="")

        missing_word.reverse()  # Reverse list to arrange in descending order of scores

        for idx, _ in missing_word:
            print(corpus.dictionary.idx2word[idx], end=", ")
        print("\n")