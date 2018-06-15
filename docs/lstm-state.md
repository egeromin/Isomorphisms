# Notes on LSTM state

Learning does affect the shape of the state changes!

Insights so far:

- more training reduces the number of required principal components / the
  'explained variance' decays more quickly
- more training makes all of the state transitions for this simple model be a
  smooth decay function, which simply 'shifts' depending on the training run.


Questions:

- is the fact that it's only so few principal components linked to the
  simplicity of the model?
- why is there this decay? and why is it so uniformly the case? Does this also
  depend on the simplicity of the model?
- what does the shift mean?
- why are currently hidden state and current state plots identical?
