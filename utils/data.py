import os
import torch
import argparse

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
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.vocab = {}
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

        self.test_left, self.test_target, self.test_right = self.tokenize_test(os.path.join(path, 'test_context_fill.txt'))
        self.context_left, self.context_right = self.tokenize_context(os.path.join(path, "context-fill.txt"))

    def preprocess(self, path):
        for fname in os.listdir(path):
            with open (os.path.join(path, fname)) as f :
                for line in f:
                    words = ['<sos>'] + line.split() + ['<eos>']
                    for word in words:
                        if word in self.vocab.keys():
                            self.vocab[word] += 1
                        else:
                            self.vocab[word] = 1

                f.close()

        c=0
        for w in self.vocab.keys():
            if self.vocab[w]>=3:
                # print(w,end=", ")
                c+=1
        print(c)
        print(len(self.vocab.keys()))

        dist=[]
        for k in self.vocab.keys():
            dist.append((self.vocab[k], k))

        dist=sorted(dist, reverse=True)
        for a,b in dist:
            print("{0}\t:\t{1}".format(b,a))


    def add_to_dict(self, path):
        assert os.path.exists(path)

        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = ['<sos>'] + line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

            return tokens

    def tokenize(self, path):
        """Tokenizes a text file."""
        tokens = self.add_to_dict(path)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = ['<sos>'] + line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

            return ids

    def tokenize_test(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            test_left=[]
            test_target=[]
            test_right=[]
            for line in f:
                ids_left = []
                ids_right = []
                words = ['<sos>'] + line.split() + ['<eos>']
                flag = False
                for word in words:
                    if word == "___":
                        flag = True
                        continue
                    if flag == False:
                        ids_left.append(self.dictionary.word2idx[word])
                    else:
                        ids_right.append(self.dictionary.word2idx[word])

                test_left.append(ids_left)
                test_target.append(self.dictionary.word2idx[f.readline().split()[0]])
                test_right.append(ids_right)

            return test_left, test_target, test_right


    def tokenize_context(self, path):
        """Tokenizes a text file."""
        self.add_to_dict(path)

        with open(path, 'r', encoding="utf8") as f:
            context_left=[]
            context_right=[]
            for line in f:
                ids_left = []
                ids_right=[]
                words = ['<sos>'] + line.split() + ['<eos>']
                flag=False
                for word in words:
                    if word == "___":
                        flag=True
                        continue
                    if flag==False:
                        ids_left.append(self.dictionary.word2idx[word])
                    else:
                        ids_right.append(self.dictionary.word2idx[word])

                context_left.append(ids_left)
                context_right.append(ids_right)

            return context_left, context_right




def main():
    parser = argparse.ArgumentParser(description='PyTorch Corpus utils')
    parser.add_argument('--data', type=str,
                        default='/home/micl-b32-24/Documents/Datasets/1-billion-word-language-modeling-benchmark/Google-1B/',
                        help='location of the data corpus')
    args = parser.parse_args()
    path = args.data
    corpus = Corpus(path)
    corpus.preprocess(path)


if __name__ == "__main__":
    main()