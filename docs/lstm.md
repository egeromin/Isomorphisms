# LSTM

Notes about getting the basic LSTM to run. 

Questions:

- what is the problem the tutorial is trying to solve?
    - it's the task of language modeling, but using 1-hot encoded vectors,
      rather than the embedding directly
- how does it differ from what I'm trying to achieve?
    - I want to use the embedding model directly and train using the cosine
      distance
- what is the model and how can it be adapted to my problem?
    - check all the above statements!

Guess of what the full model should be:

input (1-hot) -> embedding -> lstm cell -> output -> loss (x-entropy with
softmax) <- target

and the LSTM cell also depends on the *previous output* to weigh all of the
gates correctly, as well as the previous state. 

Why is there an embedding layer in the first place? Because it's more compact
than the 1-hot encoded vector, and it has this vector arithmetic property...
poor. What's the alternative? The alternative is *not* having any embedding
layer and directly pushing the input into the LSTM cell. So let's try that
first. 

In fact, that's a possibility, and it's what's done in the example. 


---


Questions about my current (buggy) code:

- what is the step size?
- I need to fix all of the parameters, which are currently random
- What input shape does the NN expect? Expected answer: `1 x size_batch`. This
  is incorrect: it should be `size_batch x sequence_length`. Reason: the input
  is a set of integers -- it *directly* replaces 1-hot encoding to give instead
  for each input index a vector of a given size. Now the Embedding layer
  computes the output for each of the input words in the sequence.
- Given this: at which point are the input vectors combined into a single
  output layer? -- this doesn't really make sense. There's just this syntax to
  make things work. LSTM 'knows' I'm using a sequence; TimeDistributed applies
  the Dense layer to each element in the sequence; Activation works out of the
  box on each temporal output.
- What is the difference between unrolling and not unrolling? Seems like
  unrolling creates actual copies of the network for each element in the
  sequence; whereas not unrolling saves space by creating a temporary cache and
  then copying during the online process? (How would this be done?)
- Which order should I feed training labels?
- Do I need to manually 1-hot encode the input vectors, or can I just provide a
  bunch of integers?


Notes about the buggy code:

- the labels need to be 1-hot encoded. Keras won't do this for us. And that
  makes sense, because the output is a vector of size `vocabulary_size`, rather
  than just an integer. 
