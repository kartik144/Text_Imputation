import os


def tokenize_file(path, dict, limit=-1, targets=False):
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

            if limit != -1 and len(words) < limit:
                words += (["<NULL>"] * (limit - len(words)))

            flag = False

            for word in words:

                if word == "___":
                    flag = True
                    continue
                if flag == False:
                    try:
                        ids_left.append(dict.word2idx[word])
                    except KeyError:
                        ids_left.append(dict.word2idx["<unk>"])
                else:
                    try:
                        ids_right.append(dict.word2idx[word])
                    except KeyError:
                        ids_right.append(dict.word2idx["<unk>"])


            left.append(ids_left)
            right.append(ids_right)

            if flag == False:
                print("## No blank inputted!! ##\n")

            if targets:
                try:
                    target.append(dict.word2idx[f.readline().split()[0]])
                except KeyError:
                    target.append(dict.word2idx["<unk>"])
        if targets:
            return left, target, right
        else:
            return left, right


def tokenize_input(sent, dict, limit=-1):
    left = []
    right = []

    words = ['<sos>'] + sent.split() + ['<eos>']

    if limit != -1 and len(words) < limit:
        words += (["<NULL>"] * (limit - len(words)))

    flag = False

    for word in words:

        if word == "___":
            flag = True
            continue
        if flag == False:
            try:
                left.append(dict.word2idx[word])
            except KeyError:
                left.append(dict.word2idx["<unk>"])
        else:
            try:
                right.append(dict.word2idx[word])
            except KeyError:
                right.append(dict.word2idx["<unk>"])

    if flag == False:
        print("## No blank inputted!! ##\n")

    return left, right