import tensorflow as tf

hello = tf.constant

hello = tf.constant('Hello, TensorFlow!')                                  
sess = tf.Session()                                               
print(sess.run(hello))

from tensorflow.contrib.rnn import BasicLSTMCell


def lstm_model():
    """
    Method to return an RNN model in tensorflow... and what is that?

    It's a class! Building a class factory.
    """
    

def build_and_train():
    """
    Build and train a model using LSTMs
    """

    # 1. Define the input. Anything, say a vector of 1s. 

    inp = tf.placeholder(tf.float32, shape=(100, 1))
    w = tf.get_variable('W', (2, 100))
    b = tf.get_variable('b', (2, 1))
    out = tf.matmul(w, inp) + b

    with tf.Session() as sess

    
