dataset_sentence_key_mapping = {
    'imdb' : 'text',
    'emotion' : 'text'
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