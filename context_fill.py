import argparse
from operator import itemgetter
import torch
import os

import context_data

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

corpus = context_data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden_left = model_left.init_hidden(1)
hidden_right = model_right.init_hidden(1)
#print("Tokens : "+str(ntokens))
with torch.no_grad():
    print("=" * 89)
    print("============================= Predicting words for test set =============================")
    print("=" * 89)
    em=0
    topV=0
    topX=0
    for index, line in enumerate(corpus.test_right):
        missing_word=[]
        input_left = torch.LongTensor(corpus.test_left[index]).view(-1,1).to(device)
        input_right = torch.LongTensor(line).view(-1,1).flip(0).to(device)


        outputs_left, hidden_left = model_left(input_left, hidden_left)
        outputs_right, hidden_right = model_left(input_right, hidden_right)

        output_flat_left = outputs_left.view(-1, ntokens)[-1]
        output_flat_right = outputs_right.view(-1, ntokens)[-1]
        output_flat = output_flat_left + output_flat_right
        #print(output_flat.size())
        #print(output_flat.size())
        #print(output_flat)

        for i in range(0,output_flat.size()[-1]):
            #print(output_flat[i].data, end=", ")
            if len(missing_word)<10:
                missing_word.append((i,output_flat[i].data))
                missing_word.sort(key=itemgetter(1))
            else:
                if output_flat[i].data > missing_word[0][1]:
                    missing_word[0]=(i,output_flat[i].data)
                    missing_word.sort(key=itemgetter(1))

        #print(missing_word[-5:])

        for w in corpus.test_left[index]:
            print(corpus.dictionary.idx2word[w],end=" ")
        print("___",end=" ")
        for w in corpus.test_right[index]:
            print(corpus.dictionary.idx2word[w],end=" ")

        print("\nTarget Word: {0}\nCandidate words: ".format(corpus.dictionary.idx2word[corpus.test_target[index]]), end= " ")

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

        output_flat_left = outputs_left.view(-1, ntokens)[-1]
        output_flat_right = outputs_right.view(-1, ntokens)[-1]
        output_flat = output_flat_left + output_flat_right
        #print(input.size())
        # outputs, hidden = model(input, hidden)
        # #print(outputs.size(),end="\t")
        # output_flat = outputs.view(-1, ntokens)[-1]
        # #print(output_flat.size())
        # #print(output_flat)

        for i in range(0,output_flat.size()[-1]):
            #print(output_flat[i].data, end=", ")
            if len(missing_word)<10:
                missing_word.append((i,output_flat[i].data))
                missing_word.sort(key=itemgetter(1))
            else:
                if output_flat[i].data > missing_word[0][1]:
                    missing_word[0]=(i,output_flat[i].data)
                    missing_word.sort(key=itemgetter(1))

        #print(missing_word[-5:])
        print(f.readline(),end="")
        print("Candidate words: ",end="")

        missing_word.reverse()  # Reverse list to arrange in descending order of scores

        for idx, _ in missing_word:
            print(corpus.dictionary.idx2word[idx], end=", ")
        print("\n")
