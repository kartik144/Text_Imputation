from operator import itemgetter


class AccuracyCounter(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def correct_(self):
        self.correct += 1
        self.total += 1

    def incorrect(self):
        self.total += 1

    def display_results(self):
        accuracy = (self.correct / self.total)*100
        print("Accuracy \t: \t{0}/{1} \t{2:2.2f}%".format(self.correct, self.total, accuracy))


def get_data(path, ans):
    sentences=[]
    with open(ans, "r") as f2:
        with open(path, "r") as f:
            while True:
                temp = f.readline()[:-1]

                if temp == '':
                    break

                options=[]
                for i in range(0,5):
                    options.append(f.readline()[:-1])

                sentences.append({'sentence': temp, 'options': options, 'answer': options[int(f2.readline()[:-1])]})

    return sentences


def get_scores(input, options, dictionary):
    words = {}
    for i in range(0, input.size(0)):
        words[dictionary.idx2word[i]] = input[i].data

    scores = []
    for w in options:
        scores.append((w,words[w]))

    scores.sort(key=itemgetter(1))
    scores.reverse()
    return scores


def get_scores_ensemble(inputs, options, dictionaries):
    Words=[]
    for index,i in enumerate(inputs):
        words = {}
        for j in range(0, i.size(0)):
            words[dictionaries[index].idx2word[j]] = i[j].data
        Words.append(words)

    scores = []
    for w in options:
        scores.append((w,0))

    for words in Words:
        for index,item in enumerate(scores):
            a,b = item
            b += words[a]
            scores[index] = (a,b)

    scores.sort(key=itemgetter(1))
    scores.reverse()

    return scores


def main():
    path = '../data/msr_test/test.txt'
    ans = '../data/msr_test/test_ans.txt'
    sentences=get_data(path,ans)
    for line in sentences:
        print(line)
    print(len(sentences))


if __name__=='__main__':
    main()