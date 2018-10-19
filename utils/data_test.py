import os
import argparse
import pickle

def tokenize(path, dict, targets = False):
    """Tokenizes a text file."""
    assert os.path.exists(path)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        left = []
        target = []
        right = []
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
                    try:
                        ids_left.append(dict.word2idx[word])
                    except:
                        ids_left.append(dict.word2idx["<unk>"])
                else:
                    try:
                        ids_right.append(dict.word2idx[word])
                    except:
                        ids_right.append(dict.word2idx["<unk>"])


            left.append(ids_left)
            right.append(ids_right)


            if targets:
                target.append(dict.word2idx[f.readline().split()[0]])

        if targets:
            return left, target, right
        else:
            return left, right