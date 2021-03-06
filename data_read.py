import os
import re
import numpy as np


import pickle
def read_movies_data(params):
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = " ".join([word for word in string.split() if len(word) > 1 and word.isalnum()])
        return string.strip().lower()

    movies = ['BMI', 'CHI', 'CRA', 'DEP', 'FNE', 'GLA', 'LOR']
    train_mv = ['BMI', 'CHI', 'FNE', 'GLA', 'LOR']
    test_mv = ['CRA', 'DEP']

    mv_idx = pickle.load(open(os.path.join(params['DATA'], "mv_idx.pkl"), "rb"))

    movie_labels = {}
    sub_data = {}
    for mv in movies:

        labels = np.loadtxt("./data/label_"+ mv +".csv", delimiter = ',')
        label = np.int64(labels[:, 1].copy())
        label = label[0:len(mv_idx[mv])]
        movie_labels[mv] = label
        x = []
        with open("./data/text_" + mv + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                x.append(line)
        x = [x[i] for i in mv_idx[mv]]
        x_cleaned = [clean_str(sent).split() for sent in x]
        sub_data[mv] = x_cleaned

    mv_data = {}
    mv_data['sub'] = {}
    mv_data['sub']['feature'] = {}
    mv_data['sub']['label'] = {}

    for mv in movies:
        mv_data['sub']['feature'][mv] = sub_data[mv]
        mv_data['sub']['label'][mv] = movie_labels[mv]

    y_train = []
    sub_train = []
    for mv in train_mv:
        y_train.extend(movie_labels[mv])
        sub_train.extend(sub_data[mv])

    y_test = []
    sub_test = []
    for mv in test_mv:
        y_test.extend(movie_labels[mv])
        sub_test.extend(sub_data[mv])

    text_data = {}
    text_data["X_train"] = sub_train
    text_data["X_test"] = sub_test
    text_data["y_train"] = y_train
    text_data["y_test"] = y_test
    return text_data


