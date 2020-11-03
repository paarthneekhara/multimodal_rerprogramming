dataset_sentence_key_mapping = {
    'imdb' : 'text',
    'emotion' : 'text',
    'ag_news' : 'text',
    'emo' : 'text',
    'yelp_polarity' : 'text'
}

dataset_num_classes = {
    'imdb' : 2,
    'emotion' : 6,
    'ag_news' : 4,
    'emo' : 4,
    'yelp_polarity' : 2
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
    else:
        pass

    return e

