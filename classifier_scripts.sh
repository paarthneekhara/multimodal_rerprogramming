python train_language_classifier.py --text_dataset imdb --language_model cnn; python train_language_classifier.py --text_dataset imdb --language_model uni_rnn; python train_language_classifier.py --text_dataset imdb --language_model bi_rnn; 

python train_language_classifier.py --text_dataset emo --language_model cnn; python train_language_classifier.py --text_dataset emo --language_model uni_rnn;  python train_language_classifier.py --text_dataset emo --language_model bi_rnn; 



python train_language_classifier.py --text_dataset ag_news --language_model cnn; python train_language_classifier.py --text_dataset ag_news --language_model bi_rnn;  python train_language_classifier.py --text_dataset ag_news --language_model uni_rnn; 


python train_language_classifier.py --text_dataset yelp_polarity --language_model cnn; python train_language_classifier.py --text_dataset yelp_polarity --language_model bi_rnn;  python train_language_classifier.py --text_dataset yelp_polarity --language_model uni_rnn; 

python train_language_classifier.py --text_dataset emotion --language_model uni_rnn; python train_language_classifier.py --text_dataset emotion --language_model bi_rnn; python train_language_classifier.py --text_dataset emotion --language_model cnn; 