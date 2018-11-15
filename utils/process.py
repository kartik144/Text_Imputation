from operator import itemgetter
from nltk.corpus import stopwords


stopWords = set(list(stopwords.words('english'))+['<eos>','<sos>', ',', ':',"\"", "?", "!","I", "A", "OK",
                                                  "_", "mr","--", "-", ")", "\'", "("])


def get_missing_word(input, dictionary, N):
    missing_word = []
    for i in range(0, input.size()[-1]):
        if (dictionary.idx2word[i].lower() in stopWords) or ('.' in dictionary.idx2word[i]):
            continue
        elif len(missing_word) < N:
            missing_word.append((i, input[i].data))
            missing_word.sort(key=itemgetter(1))
        else:
            if input[i].data > missing_word[0][1]:
                missing_word[0] = (i, input[i].data)
                missing_word.sort(key=itemgetter(1))

    return missing_word

def get_predictions(dictionary, missing_word):
    missing_word.reverse()  # Reverse list to arrange in descending order of scores
    output = []

    for idx, _ in missing_word:
        output.append(dictionary.idx2word[idx])

    return output

def print_predictions(dictionary, missing_word):
    missing_word.reverse()  # Reverse list to arrange in descending order of scores

    for idx, _ in missing_word:
        print(dictionary.idx2word[idx], end=", ")
    print()