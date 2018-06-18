"""
A character level language model in vanilla tensorflow
"""


import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
import argparse
import sys

from sklearn.decomposition import PCA

import ipdb

from lstm_state.loader import dataset_from_stage
from lstm_state import config
from lstm_state import models


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


def predict(model, inp, label, accuracy=False, return_state=False):

    state = model.zero_state()
    if return_state:
        model.start_saving_state()
    
    loss = 0
    for j in range(config.seq_length):
        prediction, state = model.forward(inp[:,j], state)

        if accuracy:
            diffs = tf.equal(tf.argmax(prediction, axis=-1),
                             tf.argmax(label[:,j], axis=-1))
            diffs = tf.cast(diffs, tf.float32)
            loss += tf.reduce_mean(diffs)
        else:
            # loss += tf.losses.sigmoid_cross_entropy(
            #             label[:,j], prediction
            #         )
            # loss += tf.reduce_mean(
            #     categorical_crossentropy(label[:,j], prediction))
            
            curr_loss = tf.losses.softmax_cross_entropy(label[:,j],
                                                    prediction)
            loss += curr_loss

    loss /= config.seq_length
            
    return loss, model.get_saved_states()


def training_loop(model, num_iterations=8000):
    optimizer = tf.train.AdamOptimizer()

    # ipdb.set_trace()
    dataset = dataset_from_stage('train')
    data_iterator = tfe.Iterator(dataset)
    
    val_dataset = dataset_from_stage('valid')
    val_iterator = tfe.Iterator(val_dataset)

    for i in range(num_iterations):
        x, y = next(data_iterator)
        
        with tfe.GradientTape() as tape:
            loss, _ = predict(model, x, y)

        grads = tape.gradient(loss, model.get_variables())
        optimizer.apply_gradients(zip(grads, model.get_variables()))

        if i % 200 == 0:
            xval, yval = next(val_iterator)
            val_loss, _ = predict(model, xval, yval, accuracy=True)
            print("Validation accuracy: {:.4f}".format(val_loss))
            print("Current loss: {:.4f}".format(loss))

            model.save(itn=i)

    return model


def generate(model):
    """
    Generate a sentence given an LSTM and a starting character
    """
    chars = []

    word = tf.zeros((1, 256), tf.float32)
        
    state = model.zero_state(batch_size=1)

    for i in range(config.seq_length):
        word, state = model.forward(word, state)
        word_prob = tf.nn.softmax(word)
        word_int = np.random.choice(256, p=word_prob.numpy()[0])
        chars.append(chr(word_int))

    return "".join(chars)


def main():

    parser = argparse.ArgumentParser(description="plot the state of an RNN "
                                     "during prediction")
    parser.add_argument("--train", help="Number of training iterations",
                        type=int, default=0)
    args = parser.parse_args()

    model = models.StackedLSTMWithDense()

    if args.train > 0:
        # model = models.LSTMWithDense()
        model = training_loop(model, args.train)
        sys.exit(0)
    else:
        model.restore()

    print("Generating a sentence using the trained LSTM")
    print(generate(model))

    dataset = dataset_from_stage('test')
    test_iterator = tfe.Iterator(dataset)

    x, y = next(test_iterator)
    print(x.shape)
    loss, states = predict(model, x, y, return_state=True, accuracy=True)
    print("Test accuracy: {:.4f}".format(loss))

    # ipdb.set_trace()

    hidden_states = np.vstack([state[0][0] for state in states])
    current_states = np.vstack([state[0][1] for state in states])

    print("Plotting variances for top 20 components")
    pca = PCA(n_components=20)
    pca.fit(current_states)
    plt.plot(pca.explained_variance_)
    plt.show()

    pca = PCA(n_components=1)
    current_states_reduced = pca.fit_transform(current_states)
    print(current_states_reduced.shape)

    pca2 = PCA(n_components=1)
    hidden_states_reduced = pca2.fit_transform(current_states)

    for _ in range(10):
        random_sentence = np.random.randint(0, config.batch_size)

        principal_state_component = [current_states_reduced[i*64 + random_sentence]
                                     for i in range(config.seq_length + 1)]
        hidden_principal_state_component = [hidden_states_reduced[i*64 + random_sentence]
                                     for i in range(config.seq_length + 1)]

        words = np.argmax(x.numpy(), axis=-1)
        print( "".join(map(chr, words[random_sentence,:])) )
        
        plt.plot(principal_state_component)
        plt.show()
        plt.plot(hidden_principal_state_component)
        plt.show()


if __name__ == "__main__":
    main()

