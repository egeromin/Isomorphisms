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



def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)



def categorical_crossentropy(target, output, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                len(output.get_shape()) - 1,
                                True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(1e-7, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output),
                               len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)


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
            # ipdb.set_trace()
            if accuracy:
                diffs = tf.equal(tf.argmax(prediction, axis=-1),
                                 tf.argmax(label[:,j], axis=-1))
                # ipdb.set_trace()
                diffs = tf.cast(diffs, tf.float32)
                loss += tf.reduce_mean(diffs)
            else:
                # loss += tf.losses.sigmoid_cross_entropy(
                #             label[:,j], prediction
                #         )
                loss += tf.reduce_mean(
                    categorical_crossentropy(label[:,j], prediction))
                # ipdb.set_trace()

        # if accuracy:
        loss /= config.seq_length
                
        return loss

    for i in range(8000):
        x, y = next(data_iterator)
        
        with tfe.GradientTape() as tape:
            loss = predict(x, y)

        if i % 25 == 0:
            xval, yval = next(val_iterator)
            val_loss = predict(xval, yval, accuracy=True)
            print("Validation accuracy: {:.4f}".format(val_loss))
            print("Current loss: {:.4f}".format(loss))
           
        grads = tape.gradient(loss, lstm.variables)
        optimizer.apply_gradients(zip(grads, lstm.variables))


def main():
    training_loop()


if __name__ == "__main__":
    main()

