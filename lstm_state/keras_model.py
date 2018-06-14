"""
Keras implementation of the model, for debugging
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import RNN, LSTM, TimeDistributed, Activation

# import ipdb

from keras import backend as K

from lstm_state.loader import dataset_from_stage
from lstm_state import config


def make_model():
    """
    Simple sequence model with only an LSTM cell and nothing else
    """
    hidden_size = 256
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True,
              input_shape=(config.seq_length, 256)))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    return model


class DataGenerator:
    """Take a tensorflow dataset object and returns a 
    generator that can be used with keras' fit_generator
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def generate(self):
        iterator = self.dataset.make_one_shot_iterator()
        # iterator = tfe.Iterator(self.dataset)

        next_data = iterator.get_next()
        while True:
            yield K.get_session().run(next_data)


def training_loop():
    train_data_generator = DataGenerator(dataset_from_stage('train'))
    valid_data_generator = DataGenerator(dataset_from_stage('valid'))
    model = make_model()
    steps_per_epoch=25
    num_epochs=200

    # ipdb.set_trace()

    model.fit_generator(train_data_generator.generate(), 
                        steps_per_epoch, num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=2,
                        workers=0)
                        # callbacks=callbacks)

def main():
    training_loop()


if __name__ == "__main__":
    main()

