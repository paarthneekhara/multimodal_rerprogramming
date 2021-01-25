import torch
import numpy as np

text_dataset_configs = {
    'imdb' : {
        'data_files' : None,
        'sentence_mapping' : 'text',
        'num_labels' : 2,
        'subset' : None,
        'val_split' : 'test'
    },
    'emotion' : {
        'data_files' : None,
        'sentence_mapping' : 'text',
        'num_labels' : 6,
        'subset' : None,
        'val_split' : 'test'
    },
    'ag_news' : {
        'data_files' : None,
        'sentence_mapping' : 'text',
        'num_labels' : 4,
        'subset' : None,
        'val_split' : 'test'
    },
    'emo' : {
        'data_files' : None,
        'sentence_mapping' : 'text',
        'num_labels' : 4,
        'subset' : None,
        'val_split' : 'test'
    },
    'yelp_polarity' : {
        'data_files' : None,
        'sentence_mapping' : 'text',
        'num_labels' : 2,
        'subset' : None,
        'val_split' : 'test'
    },
    'glue' : {
        'data_files' : None,
        'sentence_mapping' : 'sentence',
        'num_labels' : 2,
        'subset' : 'cola',
        'val_split' : 'validation'
    },
    'amazon_julian' : {
        'data_files' : {
            'train' : ['/data2/paarth/HuggingFaceDatasets/localfiles/BooksTrain.json'],
            'test' : ['/data2/paarth/HuggingFaceDatasets/localfiles/BooksTest.json']
        },
        'sentence_mapping' : 'reviewText',
        'num_labels' : 6,
        'subset' : None,
        'val_split' : 'test'

    },
    'abcd_synthetic' : {
        'data_files' : {
            'train' : ['/data2/paarth/HuggingFaceDatasets/localfiles/synthetic_abcd_train.json'],
            'test' : ['/data2/paarth/HuggingFaceDatasets/localfiles/synthetic_abcd_test.json']
        },
        'sentence_mapping' : 'sentence',
        'num_labels' : 2,
        'subset' : None,
        'val_split' : 'test'
    },
    'names' : {
        'data_files' : {
            'train' : ['/data2/paarth/HuggingFaceDatasets/localfiles/names_train.json'],
            'test' : ['/data2/paarth/HuggingFaceDatasets/localfiles/names_test.json']
        },
        'sentence_mapping' : 'sentence',
        'num_labels' : 18,
        'subset' : None,
        'val_split' : 'test'
    },
    # https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29
    'protein_splice' : {
        'data_files' : {
            'train' : ['/data2/paarth/HuggingFaceDatasets/localfiles/protein_splice_train.json'],
            'test' : ['/data2/paarth/HuggingFaceDatasets/localfiles/protein_splice_test.json']
        },
        'sentence_mapping' : 'sentence',
        'num_labels' : 3,
        'subset' : None,
        'val_split' : 'test'
    },
    'questions_correct' : {
        'data_files' : {
            'train' : ['/data2/paarth/HuggingFaceDatasets/localfiles/questions_train_correct.json'],
            'test' : ['/data2/paarth/HuggingFaceDatasets/localfiles/questions_test_correct.json']
        },
        'sentence_mapping' : 'sentence',
        'num_labels' : 6,
        'subset' : None,
        'val_split' : 'test'
    },
    'gard' : {
        'data_files' : {
            'train' : ['/data2/paarth/HuggingFaceDatasets/localfiles/gard_train.json'],
            'test' : ['/data2/paarth/HuggingFaceDatasets/localfiles/gard_test.json']
        },
        'sentence_mapping' : 'sentence',
        'num_labels' : 13,
        'subset' : None,
        'val_split' : 'test'
    },
    'goodreads_julian' : {
        'data_files' : {
            'train' : ['/data2/paarth/HuggingFaceDatasets/localfiles/goodreads_train.json'],
            'test' : ['/data2/paarth/HuggingFaceDatasets/localfiles/goodreads_test.json']
        },
        'sentence_mapping' : 'reviewText',
        'num_labels' : 2,
        'subset' : None,
        'val_split' : 'test'

    }
}


image_model_configs = {
    'vit_base_patch16_384' : {
        'mean' : (0.5, 0.5, 0.5),
        'std' : (0.5, 0.5, 0.5)
    },
    'resnet50' : {
        'mean' : (0.485, 0.456, 0.406),
        'std' : (0.229, 0.224, 0.225)
    },
    'tf_efficientnet_b7' : {
        'mean' : (0.485, 0.456, 0.406),
        'std' : (0.229, 0.224, 0.225)
    },
    'tf_efficientnet_b4' : {
        'mean' : (0.485, 0.456, 0.406),
        'std' : (0.229, 0.224, 0.225)
    },
    'inception_v3' : {
        'mean' : (0.485, 0.456, 0.406),
        'std' : (0.229, 0.224, 0.225)
    }
}

def label_mapper(e, text_dataset):
    if text_dataset == 'emotion':
        mapping = {
            'anger' : 0,
            'fear' : 1,
            'joy' : 2,
            'love' : 3,
            'sadness' : 4,
            'surprise' : 5
        }
        if isinstance(e['label'], list):
            e['label'] = [mapping[l] for l in e['label']]
        else:
            e['label'] = mapping[e['label']]
    elif text_dataset == 'amazon_julian':
        e['label'] = int(e['overall'])
    else:
        pass

    return e

class CharacterLevelTokenizer:
    def __init__(self):
        vocab = {}
        for i in range(2, 258):
            vocab[chr(i-2)] = i
        vocab["<PAD>"] = 0
        self.vocab = vocab
    
    def __call__(self, list_of_strings, truncation=True, padding="max_length", max_length=512, pad_token_id = 0):
        """
        Dont care about the truncation and padding args. just have them for backward compatibility.
        """
        attention_masks = np.zeros((len(list_of_strings), max_length), dtype=np.int64)
        input_ids = np.full((len(list_of_strings), max_length), pad_token_id, dtype=np.int64)

        for idx, string in enumerate(list_of_strings):
            # make sure string is in byte format
            if not isinstance(string, bytes):
                string = str.encode(string)

            input_ids[idx, :len(string)] = np.array([x + 2 for x in string], dtype=np.int64)
            attention_masks[idx, :len(string)] = 1

        return {
            'input_ids' : input_ids,
            'attention_mask' : attention_masks
        }

    def decode(self, outputs_ids):
        decoded_outputs = []
        for output_ids in outputs_ids.tolist():
            # transform id back to char IDs < 2 are simply transformed to ""
            decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
        return decoded_outputs
    
    def get_vocab(self):
        return self.vocab