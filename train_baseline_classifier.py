########################
# Jinglong Du
# jid020@ucsd.edu

# Sample commands to use:
#   python train_baseline_classifier.py 
#       Everything is in default
#   python train_baseline_classifier.py --text_dataset 'yelp_polarity' --Algorithm 1
#       Choose algorithm or dataset you wish 
# Unfinished:
#   check comment code to utilize the tensorboard and record the result as you wish
########################
import numpy as np
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

dataset_sentence_key_mapping = {
    'imdb' : 'text',
    'emotion' : 'text',
    'ag_news' : 'text',
    'emo' : 'text',
    'yelp_polarity' : 'text'
}


def save_obj(obj, file_path ):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--text_dataset', type=str , default = 'emotion', 
        help='Enter one of following dataset: imdb, emotion, ag_news, emo, yelp_polarity')
    p.add_argument('--algorithm', type=str, default = "sgd", 
        choices = ["logistic", "perceptron", "svc", "sgd"]
    )
    p.add_argument('--logdir', type=str, default = "/data2/paarth/ReprogrammingTransformers/BaselineClassificationModels")
    p.add_argument('--cache_dir', type=str, default = "/data2/paarth/HuggingFaceDatasets")

    args = p.parse_args()
    assert args.text_dataset in dataset_sentence_key_mapping
    


    exp_name = "baseline_{}_model_{}".format(args.text_dataset, args.algorithm)

    logdir = os.path.join(args.logdir, exp_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    subset = None
    val_split = "test"
    if args.text_dataset == "glue":
        subset = "cola"
        val_split = "validation"

    dataset = load_dataset(args.text_dataset, subset, cache_dir = args.cache_dir)

    tfidfVectorizer = TfidfVectorizer()
    tfidf = tfidfVectorizer.fit_transform(dataset['train']['text'])
    X_train = tfidf
    y_train = dataset['train']['label']

    X_test = tfidfVectorizer.transform(dataset['test']['text'])
    y_test = dataset[val_split]['label']


    classifiers = {
        "logistic" : LogisticRegression( verbose = 2, max_iter = 1000, n_jobs = -1, solver = "saga"),
        "perceptron" : Perceptron( verbose = 2),
        "svc" : LinearSVC(verbose = 2),
        "sgd" : SGDClassifier(n_jobs = -1, verbose =2)
    }

    classifier = classifiers[args.algorithm]
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    result_text = "Dataset:", args.text_dataset, "\t\tClassifier:", classifier, "\t\tScore:", score
    
    print('-------------------------------------------------')
    print(result_text)
    print('-------------------------------------------------')

    model_path = os.path.join(logdir, "model.pkl")
    save_obj(classifier, model_path)
    print("Model Saved as:", model_path)

if __name__ == '__main__':
    main()


