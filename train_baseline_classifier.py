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
    p.add_argument('--n_training', type=int, default = None)
    p.add_argument('--tfidf_analyzer', type=str, default = 'word') # set char for char-level, eg. synthetic abcd
    p.add_argument('--ngrams_min', type=int, default = 1) 
    p.add_argument('--ngrams_max', type=int, default = 1) 

    args = p.parse_args()
    
    dataset_configs = data_utils.text_dataset_configs
    assert args.text_dataset in dataset_configs
    text_dataset_config = dataset_configs[args.text_dataset]

    subset = text_dataset_config['subset']
    text_key = text_dataset_config['sentence_mapping']
    val_split = text_dataset_config['val_split']
    data_files = text_dataset_config['data_files']
    dataset_name = args.text_dataset if data_files is None else 'json'
    
    exp_name = "baseline_{}_model_{}".format(args.text_dataset, args.algorithm)

    logdir = os.path.join(args.logdir, exp_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    dataset = load_dataset(dataset_name, subset, data_files=data_files, cache_dir = args.cache_dir)
    if args.n_training is not None:
        dataset['train'] = dataset['train[0:]'.format(args.n_training)]

    ngram_range = (args.ngrams_min, args.ngrams_max)
    tfidfVectorizer = TfidfVectorizer(analyzer=args.tfidf_analyzer, ngram_range=ngram_range)
    tfidf = tfidfVectorizer.fit_transform(dataset['train'][text_key])
    X_train = tfidf
    y_train = dataset['train']['label']

    X_test = tfidfVectorizer.transform(dataset['test'][text_key])
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

