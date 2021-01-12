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
    }
}

def label_mapper(e, text_dataset):
    if text_dataset == 'emotion':
        # print("In mapper", e)
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

