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
    def __init__(self, path, threshold=1):
        self.dictionary = Dictionary()
        self.vocab = {}
        self.preprocess(path)
        self.train = self.tokenize(os.path.join(path, 'train.txt'), threshold)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), threshold)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), threshold)

        # self.test_left, self.test_target, self.test_right = self.tokenize_test(os.path.join
        # (path, 'test_context_fill.txt'))
        # self.context_left, self.context_right = self.tokenize_context(os.path.join(path, "context-fill.txt"))

    def preprocess(self, path):

        # Calculated word-frequency table to prune the vocabulary
        # depending upon the threshold

        for fname in os.listdir(path):
            with open (os.path.join(path, fname)) as f:
                for line in f:
                    words = ['<sos>'] + line.split() + ['<eos>']
                    for word in words:
                        if word in self.vocab.keys():
                            self.vocab[word] += 1
                        else:
                            self.vocab[word] = 1

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

        ############################################################################
        ############################################################################

    def tokenize(self, path, threshold):
        """Tokenizes a text file."""
        tokens = 0

        # Add to dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = ['<sos>'] + line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    if self.vocab[word] <= threshold:
                        word = "<unk>"
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = ['<sos>'] + line.split() + ['<eos>']
                for word in words:
                    if self.vocab[word] <= threshold:
                        word = "<unk>"
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

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
    parser.add_argument('--dict', type=str, default='../Dictionary/dict.pt',
                        help='path to pickled dictionary')

    args = parser.parse_args()

    corpus = Corpus(args.data, args.threshold)
    ntokens = len(corpus.dictionary)
    with open(args.dict, "wb") as f:
        pickle.dump((corpus.dictionary, args.threshold), f)

    print("Number of tokens in vocabulary: {0}".format(ntokens))

if __name__ == "__main__":
    main()