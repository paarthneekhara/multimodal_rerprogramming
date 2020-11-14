########################
# Jinglong Du
# jid020@ucsd.edu

# Sample commands to use:
#   python CodeForBaselinePY.py 
#       Everything is in default
#   python CodeForBaselinePY.py --text_dataset 'yelp_polarity' --Algorithm 1
#       Choose algorithm or dataset you wish 
# Unfinished:
#   check comment code to utilize the tensorboard and record the result as you wish
########################


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Perceptron
import time
from datasets import load_dataset
import pickle


import argparse
import data_utils
import os
# from tensorboardX import SummaryWriter


MLModelFiles = '' 

dataset_sentence_key_mapping = {
    'imdb' : 'text',
    'emotion' : 'text',
    'ag_news' : 'text',
    'emo' : 'text',
    'yelp_polarity' : 'text'
}


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--text_dataset', type=str , default = 'emotion', 
        help='Enter one of following dataset: imdb, emotion, ag_news, emo, yelp_polarity')
    p.add_argument('--Algorithm', type=int, default = 3, 
        help='Enter algorithm number 1 ~ 4: LogisticRegression, Perceptron, LinearSVC, SGDClassifier')
    # p.add_argument('--logdir', type=str, default = "/data2/paarth/ReprogrammingTransformers/ClassificationModels")
    p.add_argument('--cache_dir', type=str, default = "/data2/paarth/HuggingFaceDatasets")

    args = p.parse_args()
    assert args.text_dataset in dataset_sentence_key_mapping
    assert args.Algorithm in list(range(1, 5))


    # exp_name = "classifier_{}_model_{}".format(args.text_dataset, args.Algorithm)

    # logdir = os.path.join(args.logdir, exp_name)
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)

    # tb_writer = SummaryWriter(logdir = logdir)

    dataset = load_dataset(args.text_dataset, cache_dir = args.cache_dir)

    tfidfVectorizer = TfidfVectorizer()
    tfidf = tfidfVectorizer.fit_transform(dataset['train']['text'])
    X_train = tfidf
    y_train = dataset['train']['label']

    X_test = tfidfVectorizer.transform(dataset['test']['text'])
    y_test = dataset['test']['label']


    classifiers = [
        LogisticRegression( verbose = 2, max_iter = 1000, n_jobs = -1, solver = "saga"),
        Perceptron( verbose = 2),
        LinearSVC(verbose = 2),
        SGDClassifier(n_jobs = -1, verbose =2)
        ]

    classifier = classifiers[int(args.Algorithm)]
    
    classifier.fit(X_train, y_train)
    Score = classifier.score(X_test, y_test)

    ResultText = "Dataset:", args.text_dataset, "\t\tClassifier:", classifier, "\t\tScore:", Score
    SavePath = MLModelFiles + "yelp_polarity"+  "_" + 'SGDClassifier'+ 'Model'
    
    print('-------------------------------------------------')
    print(ResultText)
    print('-------------------------------------------------')
    # tb_writer.add_text('BaselineModel', ResultText, 0)

    print("Model Saved as:", SavePath+'.pkl')
    save_obj(classifier, SavePath)

if __name__ == '__main__':
    main()


