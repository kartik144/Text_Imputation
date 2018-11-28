import os
import torch
import argparse
import pickle

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, threshold=1, sen_lim=50, case_=False):
        self.dictionary = Dictionary()
        self.vocab = {}
        self.case = case_
        self.limit = sen_lim  # In this tokenization of file, the bptt would be self.limit
        self.max = self.preprocess(path)
        self.train = self.tokenize(os.path.join(path, 'train.txt'), threshold)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), threshold)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), threshold)

    def preprocess(self, path):

        # Calculated word-frequency table to prune the vocabulary
        # depending upon the threshold

        max=0

        for fname in os.listdir(path):
            with open (os.path.join(path, fname)) as f:
                for line in f:

                    if self.case:
                        line = line.lower()

                    words = ['<sos>'] + line.split() + ['<eos>']

                    if len(words) > self.limit:
                        continue

                    for word in words:
                        if word in self.vocab.keys():
                            self.vocab[word] += 1
                        else:
                            self.vocab[word] = 1
                    max = max if max>len(words) else len(words)

                f.close()

        ##############################################################################
        ###### Uncomment and run main() to see the word frequency table (sorted) #####
        ##############################################################################
        # dist=[]
        # for k in self.vocab.keys():
        #     dist.append((self.vocab[k], k))
        #
        # dist=sorted(dist, reverse=True)
        # for a,b in dist:
        #     print("{0}\t:\t{1}".format(b,a))
        ##############################################################################
        ##############################################################################

        return max

    def tokenize(self, path, threshold):
        """Tokenizes a text file."""
        tokens = 0

        self.dictionary.add_word("<NULL>")
        self.vocab["<NULL>"] = threshold+1
        # Add to dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:

                if self.case:
                    line = line.lower()

                words = ['<sos>'] + line.split() + ['<eos>']

                if len(words) > self.limit:
                    continue

                tokens += self.limit
                for word in words:
                    if self.vocab[word] <= threshold:
                        word = "<unk>"
                    self.dictionary.add_word(word)

        total = 0
        unk = 0
        sen = 0
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:

                if self.case:
                    line = line.lower()

                flag=0
                words = ['<sos>'] + line.split() + ['<eos>']

                if len(words) > self.limit:
                    sen += 1
                    continue

                if len(words) < self.limit :
                    words += (["<NULL>"]*(self.limit - len(words)))

                for word in words:
                    if self.vocab[word] <= threshold:
                        word = "<unk>"
                        flag = 1
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

                unk += flag
                total += 1

            print("Sentences taken in the dataset: {}/{}\t[{:.2f}%]".format(total - sen, total,
                                                                            (total - sen)*100/total))

            return ids


def main():

    parser = argparse.ArgumentParser(description='PyTorch Corpus utils')
    parser.add_argument('--data', type=str,
                        default='/home/micl-b32-24/Documents/Datasets/1-billion-word-language-modeling-benchmark/'
                                'Google-1B/',
                        help='location of the data corpus')
    parser.add_argument('--threshold', type=int,
                        default=1,
                        help='Threshold for limiting vocab size of model '
                             '(any word with frequency <= threshold will not be included)')
    parser.add_argument('--sen_length', type=int,
                        default=50,
                        help='Threshold for limiting sentences of the data '
                             '(to restrict unnecessary long sentences)')
    parser.add_argument('--dict', type=str, default='../Dictionary/dict.pt',
                        help='path to pickled dictionary')

    args = parser.parse_args()

    corpus = Corpus(args.data, args.threshold, args.sen_length)
    ntokens = len(corpus.dictionary)
    with open(args.dict, "wb") as f:
        pickle.dump((corpus.dictionary, args.threshold), f)

    print("Number of tokens in vocabulary: {0}".format(ntokens))
    print("Length of each padded sequence: {0}".format(corpus.max))

if __name__ == "__main__":
    main()