"""
Different models to try out for language modelling.
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import ipdb

from lstm_state import config


class RNNLanguageModel:

    def __init__(self):
        self.should_save_state = False
        self.saved_states = []
        self.saver = None
        self.saver_path = "checkpoints/lstm-state-ckpt"

    def set_saver_path(self, saver_path):
        self.saver_path = saver_path

    def _zero_state(self, batch_size):
        raise NotImplementedError

    def zero_state(self, batch_size=None):
        if batch_size is None:
            batch_size = config.batch_size
        return self._zero_state(batch_size)

    def get_variables(self):
        raise NotImplementedError

    def _forward(self, inp, state):
        raise NotImplementedError

    def save_state(self, state):
        raise NotImplementedError

    def forward(self, inp, state):
        output, next_state = self._forward(inp, state)
        if self.should_save_state:
            self.saved_states.append(self.save_state(next_state))
        return output, next_state

    def start_saving_state(self):
        self.should_save_state = True
        self.saved_states.append(self.save_state(self.zero_state()))

    def get_saved_states(self):
        saved_states = self.saved_states
        self.saved_states = []
        self.should_save_state = False
        return saved_states

    def save(self, itn=None):
        if self.saver is None:
            self.saver = tfe.Saver(self.get_variables())

        saver_path = self.saver_path
        if itn is not None:
            saver_path += "_{}".format(str(itn).zfill(4))
        self.saver.save(saver_path)

    def restore(self):
        if self.saver is None:
            if len(self.get_variables()) == 0:
                in_ = tf.zeros((1, 256), tf.float32)
                self.forward(in_, self.zero_state(1))
            self.saver = tfe.Saver(self.get_variables())
        self.saver.restore(self.saver_path)

    # def _gather_saveables_for_checkpoint(self):
    #     """Implementing this to checkpoint the variables
    #     I've defined"""
    #     return {
    #         x.name: x for x in self.get_variables()
    #     }


class SimpleLSTM(RNNLanguageModel):

    def __init__(self):
        super().__init__()
        self.lstm_size = 256
        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

    def _zero_state(self, batch_size):
        hidden_state = tf.zeros([batch_size, self.lstm_size])
        current_state = tf.zeros([batch_size, self.lstm_size])
        return hidden_state, current_state

    def save_state(self, state):
        return tuple(map(lambda x: x.numpy(), state))

    def get_variables(self):
        return self.lstm.variables

    def _forward(self, inp, state):
        return self.lstm(inp, state)


class LSTMWithDense(SimpleLSTM):
    
    def __init__(self):
        super().__init__()
        self.lstm_size = 400
        self.output_size = 256
        self.dense1 = tf.layers.Dense(self.lstm_size,
                                      activation=tf.nn.relu)
        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        self.dense2 = tf.layers.Dense(self.output_size)

    def get_variables(self):
        return self.dense1.variables + self.lstm.variables + \
                self.dense2.variables

    def _forward(self, inp, state):
        lstm_inp = self.dense1(inp)
        lstm_out, next_state = self.lstm(lstm_inp, state)
        out = self.dense2(lstm_out)
        return out, next_state


class StackedLSTMWithDense(RNNLanguageModel):
    """
    Stack multiple LSTM cells. 
    """

    def __init__(self, num_layers=2):
        super().__init__()
        self.lstm_size = 400
        self.output_size = 256
        self.lstm = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
             for _ in range(num_layers)])
        self.dense1 = tf.layers.Dense(self.lstm_size,
                                      activation=tf.nn.relu)
        self.dense2 = tf.layers.Dense(self.output_size)

    def _zero_state(self, batch_size):
        return self.lstm.zero_state(batch_size, tf.float32)

    def save_state(self, state):
        def get_numpy(s):
            return tuple(map(lambda x: x.numpy(), s))

        return tuple(map(get_numpy, state))

    def get_variables(self):
        return self.dense1.variables + self.lstm.variables + \
                self.dense2.variables

    def _forward(self, inp, state):
        lstm_inp = self.dense1(inp)
        lstm_out, next_state = self.lstm(lstm_inp, state)
        out = self.dense2(lstm_out)
        return out, next_state

