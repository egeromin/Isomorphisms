"""
Data generators for the image captioning project.

Provides a single generator for both the training data
and the validation data. 
"""

from PIL import Image
import numpy as np
import json
import re
from collections import defaultdict
import itertools

import numpy as np



CAPTIONS_FILE = "./data/mscoco/annotations/captions_{stage}2014.json"
IMAGES_DIR = "./data/mscoco/{stage}2014"

IMAGE_INPUT_SIZE = (244, 244, 3)
FIXED_WIDTH = 244

DICTIONARY_SIZE = 10000



def get_full_image_path(image_id, stage='train'):
    padded_image_id = str(image_id).zfill(12)

    return "{dir}/COCO_{stage}2014_{padded_id}.jpg".format(
        dir=IMAGES_DIR.format(stage=stage), padded_id=padded_image_id,
        stage=stage)


def sanitize_caption(caption):
    return re.sub('[.,]', '', caption).lower().split()


def load_image(image_id, stage='train'):
    """Returns the image corresponding to the image ID"""
    def _resize_image(image):
        resize_factor = FIXED_WIDTH / max(image.size)
        return image.resize((np.array(image.size) *
                             resize_factor).astype(np.int))

    def _pad_image(np_image):
        padded_image = np.zeros(IMAGE_INPUT_SIZE, dtype=np.uint8)
        padded_image[:np_image.shape[0],:np_image.shape[1],:] = np_image
        return padded_image

    image = Image.open(get_full_image_path(image_id, stage))
    resized_image = _resize_image(image)
    np_image = np.array(resized_image)
    if len(np_image.shape) == 2:  # sometimes we get a black-and-white image
        np_image = np.broadcast_to(np_image, (3,) + np_image.shape )
        np_image = np.moveaxis(np_image, 0, -1)
    # print(np_image.shape)
    return _pad_image(np_image)



def load_data_as_dicts(stage='train'):
    """
    Loads the input data for the given stage, where
    stage is 'train', 'test' or 'val'

    If stage is 'train', also returns a word count dictionary,
    which is then used to generate the word dictionary. 
    """
    if stage not in ('train', 'test', 'val'):
        raise RuntimeError("stage must be 'train', 'test' or 'val'")
    loaded_data = [] 
    word_count_dictionary = defaultdict(int)
    with open(CAPTIONS_FILE.format(stage=stage)) as fh:
        print("Loading captions")
        all_annotations = json.load(fh)['annotations']
        sz_training_data = len(all_annotations)
        print("Loading images")
        for i, caption_data in enumerate(all_annotations):
            image_id = caption_data['image_id']
            caption = caption_data['caption']
            sanitized_caption = sanitize_caption(caption)
            if stage == "train":
                for word in sanitized_caption:
                    word_count_dictionary[word] += 1
            image = load_image(image_id)
            loaded_data.append({'image': image, 'caption':
                                  sanitized_caption})

            if (i+1) % 1000 == 0:
                print("Loaded {} of {} images".format(i+1, sz_training_data))
                print("\n-----\nBREAKING PREMATURELY. FIX ME!\n-----\n")
                break

    return loaded_data, word_count_dictionary


def make_id_word_conversions(training_data, word_count_dictionary):
    """
    Takes the full training data and computes

        - id_from_word to id (dictionary)
        - word_from_id (list)
    """
    most_frequent_words = sorted(word_count_dictionary.items(), key=lambda x: x[1])
    most_frequent_words = [x[0] for x in most_frequent_words][:DICTIONARY_SIZE
                                                              - 2]
    most_frequent_words.extend(['.', '<unk>'])

    # Here, we add a default to return the index of
    # "<unk>", which is the last element of most_frequent_words
    id_from_word = defaultdict( lambda: len(most_frequent_words) - 1, dict(zip(most_frequent_words,
                                         range(DICTIONARY_SIZE))))
    word_from_id = {v:k for k,v in id_from_word.items()}
    return id_from_word, word_from_id


def tokens_from_data(input_data, id_from_word):
    """
    Given input data and id_from_word, make the captions for each data
    item by populating the `tokenized_caption` attribute.
    """
    for item in input_data:
        item['tokenized_caption'] = [id_from_word[word] for word in
                                     item['caption']]


def to_1_hot(inp, vocabulary_size):
    """
    Convert a matrix of labels into a 1-hot
    encoded matrix. Increase dimensions along the
    last axis.
    """
    dims = inp.shape + (vocabulary_size,)
    out = np.zeros(dims)
    for i, x in np.ndenumerate(inp):
        out[i + (x,)] = 1
    return out


def main():
    loaded_data, word_dictionary = load_data_as_dicts(stage='train')
    id_from_word, word_from_id = make_id_word_conversions(loaded_data,
                                                          word_dictionary)
    tokens_from_data(loaded_data, id_from_word)
    for item in itertools.islice(loaded_data, 10):
        print(item['tokenized_caption'])    

if __name__ == "__main__":
    main()

