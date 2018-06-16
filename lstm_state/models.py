"""
Different models to try out for language modelling.
"""
import tensorflow as tf

from lstm_state import config


class RNNLanguageModel:

    def __init__(self):
        self.should_save_state = False
        self.saved_states = []

    def zero_state(self):
        raise NotImplementedError

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


class SimpleLSTM(RNNLanguageModel):

    def __init__(self):
        self.lstm_size = 256
        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        super().__init__()

    def zero_state(self):
        hidden_state = tf.zeros([config.batch_size, self.lstm_size])
        current_state = tf.zeros([config.batch_size, self.lstm_size])
        return hidden_state, current_state

    def save_state(self, state):
        return tuple(map(lambda x: x.numpy(), state))

    def get_variables(self):
        return self.lstm.variables

    def _forward(self, inp, state):
        return self.lstm(inp, state)

