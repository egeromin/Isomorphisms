"""
Loader for the tensorflow data
"""
import tensorflow as tf
import numpy as np

from lstm_state import config
from lstm_state.convert_to_tfrecord import get_output_path


def dataset_from_stage(stage, batch=True, shuffle=True):

    def parse_tfexample(example_proto):
        features = {
            'input': tf.FixedLenFeature(shape=[], dtype=tf.string), 
        }
        parsed_features = tf.parse_single_example(
            serialized=example_proto, features=features)

        full_text = tf.decode_raw(parsed_features['input'], out_type=tf.uint8)

        x = tf.one_hot(full_text[:-1], depth=256)
        y = tf.one_hot(full_text[1:], depth=256)

        return x, y
    
    pattern_tfrecord = get_output_path(stage)
    dataset = tf.data.TFRecordDataset.list_files(
        file_pattern=pattern_tfrecord
    ).interleave(tf.data.TFRecordDataset, cycle_length=config.num_tfrecords,
                 block_length=1)
    dataset = dataset.map(parse_tfexample)
    if batch:
        dataset = dataset.batch(config.batch_size)
    if shuffle:
        dataset = dataset.shuffle(10000)
    return dataset


def main():
    print("Testing the loader. Check if the result looks OK")
    import tensorflow.contrib.eager as tfe
    tf.enable_eager_execution()

    dataset = dataset_from_stage("train", batch=False)

    for _ in range(10):
        x, y = next(tfe.Iterator(dataset))

        def to_string(one_hot_tensor):
            arr = np.argmax(one_hot_tensor.numpy(), axis=1)
            return "".join(map(chr, arr))

        print(list(map(to_string, (x, y))))


if __name__ == "__main__":
    main()

