"""
A character level language model in vanilla tensorflow
"""


import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

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


def predict(lstm, inp, label, accuracy=False, return_state=False):
    hidden_state = tf.zeros([config.batch_size, lstm_size])
    current_state = tf.zeros([config.batch_size, lstm_size])

    state = hidden_state, current_state

    states = []
    if return_state:
        states.append(tuple(map(lambda x: x.numpy(), state)))
    
    loss = 0
    for j in range(config.seq_length):
        prediction, state = lstm(inp[:,j], state)
        if return_state:
            states.append(tuple(map(lambda x: x.numpy(), state)))
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
            
    return loss, states


def training_loop(num_iterations=8000):
    optimizer = tf.train.AdamOptimizer()
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # ipdb.set_trace()
    dataset = dataset_from_stage('train')
    data_iterator = tfe.Iterator(dataset)
    
    val_dataset = dataset_from_stage('valid')
    val_iterator = tfe.Iterator(val_dataset)


    for i in range(num_iterations):
        x, y = next(data_iterator)
        
        with tfe.GradientTape() as tape:
            loss, _ = predict(lstm, x, y)

        if i % 25 == 0:
            xval, yval = next(val_iterator)
            val_loss, _ = predict(lstm, xval, yval, accuracy=True)
            print("Validation accuracy: {:.4f}".format(val_loss))
            print("Current loss: {:.4f}".format(loss))
           
        grads = tape.gradient(loss, lstm.variables)
        optimizer.apply_gradients(zip(grads, lstm.variables))

    return lstm


def generate(lstm, start='a'):
    """
    Generate a sentence given an LSTM and a starting character
    """
    chars = [start]
    start = ord(start)

    # ipdb.set_trace()

    word = tf.one_hot(tf.convert_to_tensor([start]), depth=256)
        
    hidden_state = tf.zeros([1, lstm_size])
    current_state = tf.zeros([1, lstm_size])

    state = hidden_state, current_state

    for i in range(config.seq_length):
        word, state = lstm(word, state)
        word_int = tf.argmax(word, axis=-1)
        chars.append(chr(word_int.numpy()[0]))

    return "".join(chars)



def main():
    lstm = training_loop(200)
    print("Generating a sentence using the trained LSTM")

    print(generate(lstm))


    dataset = dataset_from_stage('test')
    test_iterator = tfe.Iterator(dataset)

    x, y = next(test_iterator)
    print(x.shape)
    _, states = predict(lstm, x, y, return_state=True)

    # ipdb.set_trace()

    hidden_states = np.vstack([state[0] for state in states])
    current_states = np.vstack([state[1] for state in states])

    print("Plotting variances for top 20 components")
    pca = PCA(n_components=20)
    pca.fit(current_states)
    plt.plot(pca.explained_variance_)
    plt.show()

    pca = PCA(n_components=1)
    current_states_reduced = pca.fit_transform(current_states)
    print(current_states_reduced.shape)

    for _ in range(10):
        random_sentence = np.random.randint(0, config.batch_size)

        principal_state_component = [current_states_reduced[i*64 + random_sentence]
                                     for i in range(config.seq_length + 1)]

        words = np.argmax(x.numpy(), axis=-1)
        print( "".join(map(chr, words[random_sentence,:])) )
        
        plt.plot(principal_state_component)
        plt.show()




if __name__ == "__main__":
    main()

