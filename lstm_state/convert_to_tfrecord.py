"""Convert the wikipedia dataset to tfrecord"""

import os
import argparse
import tensorflow as tf
import re
from unidecode import unidecode
# used for converting to ascii


from lstm_state import config


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_single_record(tfrecord, data):
    """
    Write a single item of data to the file. 
    """
    feature = {
        'input': _bytes_feature(data.encode('ascii'))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    tfrecord.write(example.SerializeToString())


def get_character_stream(stage='train', chunk=1):
    data_path = os.path.join(config.data_dir, "wiki.{stage}.raw{chunk}").format(
        stage=stage, chunk=str(chunk).zfill(2))
    return open(data_path, 'r'), os.path.getsize(data_path)


def get_output_path(stage, chunk="*"):
    if chunk != "*":
        chunk = str(chunk).zfill(2)

    return os.path.join(
        config.data_dir, "{stage}.{chunk}.tfrecord".format(
            stage=stage, chunk=chunk))


class DataBuffer:
    """
    A class for streaming data
    """

    def __init__(self, char_stream):
        self.char_stream = char_stream
        self.data = ""
        self.refresh()

    def refresh(self):
        """
        Fetch the next config.seq_length number of characters, and return
        the excess characters consumed in the stream as well. 
        """
        self.data = self.data[config.seq_length:]
        next_utf8 = self.char_stream.read(1 + config.seq_length -
                                          len(self.data))
        # ensure there's at least 1 + config.seq_length items in self.data
        next_ascii = unidecode(next_utf8)
        next_ascii = re.sub(r'\s+', ' ', next_ascii).lower()

        self.data = self.data + next_ascii
        if len(self.data) < 1 + config.seq_length:
            self.data = ""


def to_tfrecord(stage='train', limit=None, path_tfrecord=None,
                chunk=1):
    """
    Convert a specific stage to tfrecord. If the stage is train, it also
    computes the id_from_word and word_from_id dictionaries; otherwise,
    it attempts to read these from file.
    """

    char_stream, num_bytes = get_character_stream(stage, chunk)

    if path_tfrecord is None:
        path_tfrecord = get_output_path(stage, chunk)

    with tf.python_io.TFRecordWriter(path_tfrecord) as tfrecord:

        buf = DataBuffer(char_stream)
        approximate_bytes_processed = 0  # just for displaying progress

        while len(buf.data) > 0:

            write_single_record(tfrecord, buf.data)
            buf.refresh()

            # the rest is just for displaying progress
            approximate_bytes_processed += config.seq_length
            if approximate_bytes_processed % 10000 == 0:
                print("Processed about {}%%".format(
                    approximate_bytes_processed / num_bytes))


def main():
    parser = argparse.ArgumentParser(description="Convert raw text into a "
                                     "bunch of tfrecords")
    parser.add_argument("--stage", help="Stage: train/valid/test",
                        default="train")
    parser.add_argument("--chunk", help="Chunk number",
                        type=int, default=1)
    args = parser.parse_args()

    to_tfrecord(stage=args.stage, chunk=args.chunk)


if __name__ == "__main__":
    main()

