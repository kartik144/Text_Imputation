import argparse
from nltk.corpus import stopwords
import random
import os

parser = argparse.ArgumentParser(description='PyTorch Context-filling Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
args = parser.parse_args()
stopWords = set(stopwords.words('english'))

saveFile = open(os.path.join(args.data, "test_context_fill.txt"), "w")

with open(os.path.join(args.data, "test.txt")) as f:
    for line in f:
        words = line.split()
        rand = random.randint(0, len(words)-1)

        while words[rand] in stopWords:
            rand = random.randint(0, len(words)-1)

        for w in words[0:rand]:
            saveFile.write(w)
            saveFile.write(" ")

        saveFile.write("___")

        for w in words[rand+1:]:
            saveFile.write(" ")
            saveFile.write(w)

        saveFile.write("\n")
        saveFile.write(words[rand])
saveFile.write("\n")