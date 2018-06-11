"""
A character level language model in vanilla tensorflow
"""


import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

import ipdb

from lstm_state.loader import dataset_from_stage
from lstm_state import config

tf.enable_eager_execution()


lstm_size = 256


def training_loop():
    optimizer = tf.train.AdamOptimizer()
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # ipdb.set_trace()
    dataset = dataset_from_stage('train')
    data_iterator = tfe.Iterator(dataset)
    
    val_dataset = dataset_from_stage('valid')
    val_iterator = tfe.Iterator(val_dataset)

    def predict(inp, label, accuracy=False):
        hidden_state = tf.zeros([config.batch_size, lstm_size])
        current_state = tf.zeros([config.batch_size, lstm_size])
        state = hidden_state, current_state
        loss = 0
        for j in range(config.seq_length):
            prediction, state = lstm(inp[:,j], state)
            if accuracy:
                diffs = (tf.argmax(prediction, axis=-1) ==
                         tf.argmax(label[:,j], axis=-1))
                diffs = tf.cast(diffs, tf.float32)
                loss += tf.reduce_mean(diffs)
            else:
                loss += tf.losses.sigmoid_cross_entropy(
                            prediction, label[:,j]
                        )

        if accuracy:
            loss /= config.seq_length
                
        return loss

    for i in range(8000):
        x, y = next(data_iterator)
        
        with tfe.GradientTape() as tape:
            loss = predict(x, y)

        if i % 25 == 0:
            xval, yval = next(val_iterator)
            val_loss = predict(xval, yval, accuracy=True)
            print("Validation accuracy: {:.2f}".format(val_loss))
        if True:
            print("Current loss: {:.2f}".format(loss))
           
        grads = tape.gradient(loss, lstm.variables)
        optimizer.apply_gradients(zip(grads, lstm.variables))


def main():
    training_loop()


if __name__ == "__main__":
    main()

