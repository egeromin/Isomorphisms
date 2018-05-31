import tensorflow as tf
import keras
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3

from keras.models import Sequential
from keras.layers import Input, Embedding, RNN, LSTM, TimeDistributed, Activation, Dropout, Dense, Layer
from keras.callbacks import ModelCheckpoint
# import ipdb
import argparse

from image_captioning.convert_to_tfrecord import parse_tfrecord
from image_captioning import config
from image_captioning.data import make_data_generator
from image_captioning.convert_to_tfrecord import load_conversions

 #tf.enable_eager_execution()



class ExpandDims(Layer):
    """Custom layer for expanding dims along a given axis"""

    def __init__(self, axis, **kwargs):
        self.axis=axis
        super().__init__(**kwargs)

    def call(self, input_tensor):
        return tf.expand_dims(input_tensor, axis=self.axis)

    def compute_output_shape(self, input_shape):
            return input_shape[:1] + (1,) + input_shape[1:]


def image_captioning_model(seq_length, hidden_size, vocabulary_size):
    image = Input(shape=(299, 299, 3))
    inception_model = InceptionV3(weights='imagenet')

    text_labels = Input(shape=(seq_length,))

    embedding_layer = Embedding(vocabulary_size, hidden_size,
                                input_length=seq_length)
    embedding = embedding_layer(text_labels)

    lstm = LSTM(hidden_size, return_sequences=True)

    image_classes = inception_model(image)

    
    # In order to change the dimensions of `image_classes`, we
    # define a custom layer `ExpandDims`, because
    # using `expand_dims` from TF directly returns a tensor where a Keras
    # layer output is required.
    image_classes_expanded = ExpandDims(1)(image_classes)

    embedding_with_input_image = \
        keras.layers.concatenate([image_classes_expanded, embedding], axis=1)

    # We are not using a special start word, or designating
    # the start word any differently. Instead, we feed the image
    # classes directly into the LSTM as the first input in the
    # sequence by adjusting the shape of the embedding vector.
    # TODO:  Look into why a start word is necessary
    
    lstm_outputs = lstm(embedding_with_input_image)
    dropout_layer = Dropout(0.5)
    regularized_lstm_outputs = dropout_layer(lstm_outputs)

    fc_layer = TimeDistributed(Dense(vocabulary_size))
    fc_activation = Activation('softmax')

    outputs = fc_activation(fc_layer(regularized_lstm_outputs))

    model = Model(inputs=[image, text_labels], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    return model


def train(model, train_data_generator, valid_data_generator):
    model.fit_generator(train_data_generator.generate(),
                        100, 45,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=10,
                        workers=0)

    


def main():
    parser = argparse.ArgumentParser(description="train the image captioning "
                                     "pipeline")
    parser.add_argument("--concurrency", help="Num of threads to read data in "
                        "concurrently", type=int, default=1)

    args = parser.parse_args()

    train_data_generator = make_data_generator(stage='train',
                                               num_chunks=args.concurrency)
    valid_data_generator = make_data_generator(stage='valid',
                                               num_chunks=args.concurrency)

    word_from_id, id_from_word, seq_length = load_conversions()
    vocabulary_size = len(word_from_id)

    print("seq_length = {}".format(seq_length))

    model = image_captioning_model(seq_length, 1000,
                                   vocabulary_size)

    train(model, train_data_generator, valid_data_generator)


if __name__ == "__main__":
    main()

