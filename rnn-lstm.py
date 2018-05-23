from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, RNN, LSTM, TimeDistributed, Activation, Dropout, Dense
from keras.callbacks import ModelCheckpoint
import keras
import random

import tensorflow as tf

import numpy as np
from itertools import chain
from collections import defaultdict
import argparse


# tf.enable_eager_execution()


# reference: http://adventuresinmachinelearning.com/keras-lstm-tutorial/

# this is a variant of the lstm.py demo, but without the lstm cell, only the
# basic rnn cell. 


class SimpleRNNCell:

    # see https://keras.io/layers/recurrent/
    # and specifically the documentation for the input cell:
    # cell: A RNN cell instance. A RNN cell is a class that has:
#         a call(input_at_t, states_at_t) method, returning (output_at_t,
#                                                            states_at_t_plus_1).
#         The call method of the cell can also take the optional argument
#         constants, see section "Note on passing external constants" below.
#         a state_size attribute. This can be a single integer (single state) in
#         which case it is the size of the recurrent state (which should be the
#                                                           same as the size of
#                                                           the cell output).
#         This can also be a list/tuple of integers (one size per state). In this
#         case, the first entry (state_size[0]) should be the same as the size of
#         the cell output. It is also possible for cell to be a list of RNN cell
#         instances, in which cases the cells get stacked on after the other in
#         the RNN, implementing an efficient stacked RNN.

    def __init__(self, state_size, input_size, output_size):
        self.state_size = state_size
        self.W = tf.get_variable("W", [state_size, state_size])
        self.U = tf.get_variable("U", [input_size, state_size])
        self.V = tf.get_variable("V", [state_size, output_size])
        self.b = tf.get_variable("b", [state_size],
                                 initializer=tf.zeros_initializer)
        self.c = tf.get_variable("c", [output_size],
                                 initializer=tf.zeros_initializer)
        self.state_activation = tf.tanh  # hardcode tanh

    def call(self, input_at_t, states_at_t):
        state = states_at_t[0]
        a = tf.matmul(input_at_t, self.U) + tf.matmul(state, self.W) +  self.b
        next_state = self.state_activation(a)
        output_at_t = tf.matmul(next_state, self.V) + self.c
        return (output_at_t, [next_state])

    def get_config(self):
        return {}


def make_model(vocabulary_size, hidden_size, num_steps, use_dropout=True,
               lstm=False):
    model = Sequential()
    model.add(Embedding(vocabulary_size, hidden_size, input_length=num_steps))
    
    if lstm:
        model.add(LSTM(hidden_size, return_sequences=True))
        model.add(LSTM(hidden_size, return_sequences=True))
    else:
        model.add(RNN(SimpleRNNCell(hidden_size, hidden_size, hidden_size), return_sequences=True))
    
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocabulary_size)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    return model

def load_data():
    return training_generator, validation_generator, vocabulary_size


def ids_and_index_from_ptb(ptb_path, eos=True):
    """
    Reads a PTB file and returns:

        - id_from_word to id (dictionary)
        - word_from_id (list)
        - data (list), a long list of the full data. 

    If EOS is true, then add end of sentence tokens. 
    """
    with open(ptb_path) as fh:
        data_str = fh.read()
    
    sentences = data_str.split('\n')

    def get_sentences():
        for sentence in sentences:
            if eos:
                yield sentence.split() + ['<eos>']
            else:
                yield sentence.split()

    words = chain.from_iterable(get_sentences())
    # why are end of sentence tokens necessary?

    class WordCounter:
        
        def __init__(self):
            self.count = -1
            
        def __call__(self):
            self.count += 1
            return self.count
    
    counter = WordCounter()
    id_from_word = defaultdict(counter)

    data = [id_from_word[word] for word in words]
    word_from_id = {v: k for k, v in id_from_word.items()}  # would be nice to
    # do this using WordCounter

    return id_from_word, word_from_id, data


def ids_from_ptb(ptb_path, id_from_word, eos=True):

    with open(ptb_path) as fh:
        data_str = fh.read()
    
    sentences = data_str.split('\n')

    def get_sentences():
        for sentence in sentences:
            if eos:
                yield sentence.split() + ['<eos>']
            else:
                yield sentence.split()

    words = chain.from_iterable(get_sentences())
    data = [id_from_word[word] for word in words]
    return data


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


class DataWithLabelGenerator:

    def __init__(self, data, step_size, batch_size, vocabulary_size):
        self.data = data
        self.step_size = step_size
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        stride_size = self.batch_size * self.step_size

        # if len(self.data) % stride_size == 0:
        #     raise NotImplementedError("This case hasn't been implemented yet")
        # else:
        #     print(len(self.data) % self.step_size)
        #     print(len(self.data) % self.batch_size)

    def generate(self):
        """
        Generate training / validation data given the full data ids.

        Outputs a list of IDs in batches. 

        Question: what are the dimensions of this data?

        num_time_step * batch_size?
        """

        position = 0

        # shuffling is not implemented
        # TODO: implement shuffling.

        stride_size = self.batch_size * self.step_size
        len_data = len(self.data)

        len_data = len_data - (len_data % self.step_size)
        self.data = self.data[:len_data]

        while True:
            if position >= len(self.data) - self.step_size:
                position = 0  # reset the training data if I'm at the end. 

            if position+stride_size+1 >= len(self.data):
                len_data_m = len_data - 1
                end = len_data_m - (len_data_m % self.step_size)

            else:
                end = position+stride_size
                
            x = self.data[position:end]
            y = self.data[position+1:end+1]

            position = end

            x = np.array(x).reshape(-1, self.step_size)
            y = np.array(y).reshape(-1, self.step_size)
            y = to_1_hot(y, self.vocabulary_size)

            yield x, y


def run_training(model, train_data_generator, steps_per_epoch, num_epochs,
                 valid_steps_per_epoch, valid_data_generator, lstm=False,
                 no_checkpoints=False):
    checkpoints_path = "./checkpoints"
    callbacks=[]
    if not no_checkpoints:
        if lstm:
            checkpointer = ModelCheckpoint(filepath=checkpoints_path +
                                           '/model-lstm-{epoch:02d}.hdf5', verbose=1)
        else:
            checkpointer = ModelCheckpoint(filepath=checkpoints_path +
                                           '/model-simplernn-{epoch:02d}.hdf5', verbose=1)
        callbacks.append(checkpointer)

    if lstm:
        tensorboard_logs = "./logs-lstm"
    else:
        tensorboard_logs = "./logs-rnn"

    tboard = keras.callbacks.TensorBoard(log_dir=tensorboard_logs)
    callbacks.append(tboard)
    model.fit_generator(train_data_generator.generate(), 
                        # len(train_data)//(batch_size*num_steps), num_epochs,
                        steps_per_epoch, num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=valid_steps_per_epoch,
                        callbacks=callbacks)


def make_prediction(model, seed_sentences, word_from_id, seq_length=10):

    num_sentences, seed_sentence_length = seed_sentences.shape
    assert(seq_length > seed_sentence_length)

    predicted_sentence_indices = np.zeros((num_sentences, seq_length),
                                          dtype=np.int32)

    predicted_sentence_indices[:,:seed_sentence_length] = seed_sentences

    for i in range(seq_length-seed_sentence_length-1):
        words = predicted_sentence_indices[:,i:seed_sentence_length+i]
        next_words = model.predict(words)  # output is 1-hot encoded
        assert(len(next_words.shape) == 3)
        assert(next_words.shape[0:2] == words.shape)

        predicted_sentence_indices[:,seed_sentence_length+i+1] = \
                next_words[:,-1,:].argmax(axis=-1)

    # word ids to sentences
    predicted_sentences = []
    for i in range(predicted_sentence_indices.shape[0]):
        predicted_sentences.append([])
        for j in range(predicted_sentence_indices.shape[1]):
            word_id = predicted_sentence_indices[i,j]
            predicted_sentences[-1].append(word_from_id[word_id])

    return predicted_sentences


def load_model():
    path_model = "./checkpoints/final.hdf5"
    model = keras.models.load_model(path_model)
    return model


def random_predict(path_predict, step_size=5, seq_length=20):
    data_dir = "./data/ptb/data/"
    path_train = data_dir + "ptb.train.txt"
    id_from_word, word_from_id, train_data = ids_and_index_from_ptb(path_train,
                                                                    eos=True)
    num_sentences = 10
    seed_sentence_length = step_size

    # pick `num_sentences` worth of start indexes and use those to 
    # generate some seed sentences

    start_indices = [random.randint(0, len(train_data)) 
                     for _ in range(num_sentences)]
    seed_sentences = np.zeros((num_sentences, seed_sentence_length),
                              dtype=np.int32)
    
    for i in range(num_sentences):
        words = [train_data[j+start_indices[i]]
                 for j in range(seed_sentence_length)]
        seed_sentences[i,:] = words

    assert(seed_sentences.max() < max(id_from_word.values()))
    model = load_model()

    predicted_sentences = make_prediction(model, seed_sentences, word_from_id, seq_length)

    with open(path_predict, "w") as fh:
        for sentence in predicted_sentences:
            fh.write(" ".join(sentence))
            fh.write("\n")


def main(debug=False, lstm=False, num_epochs=1, steps_per_epoch=None,
         valid_steps_per_epoch=None, no_checkpoints=False, step_size=5):
    data_dir = "./data/ptb/data/"
    path_train = data_dir + "ptb.train.txt"
    path_valid = data_dir + "ptb.valid.txt"

    
    id_from_word, word_from_id, train_data = ids_and_index_from_ptb(path_train, eos=True)


    valid_data = ids_from_ptb(path_valid, id_from_word, eos=True)

    if debug:
        train_data = train_data[:len(train_data)//100]

    train_data_size = len(train_data)
    valid_data_size = len(valid_data)
    vocabulary_size = len(id_from_word)
    hidden_size = 128

    batch_size = 32

    # print(train_data_size)
    # print(valid_data_size)

    if steps_per_epoch is None:
        steps_per_epoch = train_data_size // (batch_size * step_size)
    
    if valid_steps_per_epoch is None:
        valid_steps_per_epoch = valid_data_size // (batch_size * step_size)

    train_data_generator = DataWithLabelGenerator(train_data, step_size,
                                                  batch_size, vocabulary_size)
    valid_data_generator = DataWithLabelGenerator(valid_data, step_size,
                                                  batch_size, vocabulary_size)
    model = make_model(vocabulary_size, hidden_size, step_size,
                       use_dropout=True, lstm=lstm)
    try:
        run_training(model, train_data_generator, steps_per_epoch, num_epochs,
                     valid_steps_per_epoch, valid_data_generator, lstm,
                     no_checkpoints=no_checkpoints)
    except KeyboardInterrupt:
        print("Ending prematurely")

    path_final_model = "./checkpoints/final.hdf5"
    model.save(path_final_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNN for the Penn Tree Bank"
                                                 "dataset")
    parser.add_argument("--debug", help="Run in 'debug' mode, i.e. with only"
                                        "1/100th of the training data", 
                        action="store_true")
    parser.add_argument("--lstm", help="Use lstm?", action='store_true')
    parser.add_argument("--num_epochs", help="Number of training epochs",
                        type=int, default=1)
    parser.add_argument("--vsteps", help="Validation steps per epoch",
                        type=int)
    parser.add_argument("--steps", help="Training steps per epoch",
                        type=int)
    parser.add_argument("--no_checkpoints", help="Ignore checkpoints",
                        action='store_true')

    parser.add_argument("--pred", help="Make some random predictions")
    parser.add_argument("--step_size", help="Step size for training and"
                        "prediction", type=int, default=5)

    args = parser.parse_args()

    if args.pred:
        random_predict(args.pred, step_size=args.step_size)
    else:
        main(args.debug, lstm=args.lstm, num_epochs=args.num_epochs,
             steps_per_epoch=args.steps, valid_steps_per_epoch=args.vsteps,
             no_checkpoints=args.no_checkpoints, step_size=args.step_size)
         


