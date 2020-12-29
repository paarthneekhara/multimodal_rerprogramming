########################
# Jinglong Du
# jid020@ucsd.edu

# Sample commands to use:
#   python train_baseline_classifier.py 
#       Everything is in default
#   python train_baseline_classifier.py --text_dataset yelp_polarity --algorithm svc
#       Choose algorithm or dataset you wish 
#    
#    python train_baseline_classifier.py --algorithm svc --deepyeti_dataset Musical_Instruments.json.gz
#       Use deepyeti's Amazon Review Dataset in .gz format.
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

import pandas as pd
import gzip
import json

import argparse
import data_utils
import os
# from tensorboardX import SummaryWriter

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def save_obj(obj, file_path ):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def ValidReview(reviewTextList, overallList):
	NewreviewTextList = []
	NewoverallList = []

	for i in range(len(reviewTextList)):
		if isinstance(reviewTextList[i], str):
			if len(reviewTextList[i])> 0:
				NewreviewTextList.append(reviewTextList[i])
				NewoverallList.append(overallList[i])

	return NewreviewTextList, NewoverallList


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--text_dataset', type=str , default = 'emotion', 
        help='Enter one of following dataset: imdb, emotion, ag_news, emo, yelp_polarity')
    p.add_argument('--algorithm', type=str, default = "sgd", 
        choices = ["logistic", "perceptron", "svc", "sgd"]
    )
    p.add_argument('--logdir', type=str, default = "/data2/paarth/ReprogrammingTransformers/BaselineClassificationModels")
    p.add_argument('--cache_dir', type=str, default = "/data2/paarth/HuggingFaceDatasets")
    helpText = "Please enter the path of custom dataset, like: Musical_Instruments.csv. It should be in format of at least two colums: 'Review' and 'rate'. rate should be integer from 1 to 5."
    p.add_argument('--custom_dataset', type=str, help = helpText )
    p.add_argument('--deepyeti_dataset' , type=str, help = "Please enter the path of deepyeti's dataset, like: Musical_Instruments.json.gz")



    args = p.parse_args()

    logdir = os.path.join(args.logdir, exp_name)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    if args.custom_dataset:
        # Not ready yet
        DataSetName = args.custom_dataset
        exit()

    elif args.deepyeti_dataset:
        df = getDF(args.deepyeti_dataset)
        reviewTextList  = df['reviewText'].tolist()
        overallList  = df['overall'].tolist()
        reviewTextList, overallList = ValidReview(reviewTextList, overallList)

        datasetSize = len(reviewTextList)
        print("Dataset Size is: ", datasetSize)

        DataSpliter = int(0.8*datasetSize)
        tfidfVectorizer = TfidfVectorizer()

        tfidf = tfidfVectorizer.fit_transform(reviewTextList[:DataSpliter])
        X_train = tfidf
        y_train = overallList[:DataSpliter]

        X_test = tfidfVectorizer.transform(reviewTextList[DataSpliter:])
        y_test = overallList[DataSpliter:]
        DataSetName = args.deepyeti_dataset
    else: # process as normal
        dataset_configs = data_utils.text_dataset_configs
        print(args.text_dataset)
        print(dataset_configs[args.text_dataset])

        assert args.text_dataset in dataset_configs
        text_dataset_config = dataset_configs[args.text_dataset]

        subset = text_dataset_config['subset']
        text_key = text_dataset_config['sentence_mapping']
        val_split = text_dataset_config['val_split']
        data_files = text_dataset_config['data_files']
        dataset_name = args.text_dataset if data_files is None else 'json'
        
        exp_name = "baseline_{}_model_{}".format(args.text_dataset, args.algorithm)

        dataset = load_dataset(dataset_name, subset, data_files=data_files, cache_dir = args.cache_dir)

        tfidfVectorizer = TfidfVectorizer()
        tfidf = tfidfVectorizer.fit_transform(dataset['train'][text_key])
        X_train = tfidf
        y_train = dataset['train']['label']

        X_test = tfidfVectorizer.transform(dataset['test'][text_key])
        y_test = dataset[val_split]['label']
        DataSetName = args.text_dataset

    classifiers = {
        "logistic" : LogisticRegression( verbose = 2, max_iter = 1000, n_jobs = -1, solver = "saga"),
        "perceptron" : Perceptron( verbose = 2),
        "svc" : LinearSVC(verbose = 2),
        "sgd" : SGDClassifier(n_jobs = -1, verbose =2)
    }

    classifier = classifiers[args.algorithm]
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    result_text = "Dataset:", DataSetName, "\t\tClassifier:", classifier, "\t\tScore:", score
    
    print('-------------------------------------------------')
    print(result_text)
    print('-------------------------------------------------')

    model_path = os.path.join(logdir, "model.pkl")
    save_obj(classifier, model_path)
    print("Model Saved as:", model_path)

if __name__ == '__main__':
    main()


