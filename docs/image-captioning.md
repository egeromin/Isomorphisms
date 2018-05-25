# Image Captioning

Todo:

1. Find and understand dataset
1. Build data pipeline for training and validation with a mock model.
    * resize the images. In the vinyals implementation, first the images are
      resized to 346 * 346, and then a random crop is taken.  Is this the same
      that's done when training InceptionV3?
    * batch the training data
    * compute tags for each of the words (complete dictionary)
    * one-hot encoding for the words
1. Substitute the correct model, finding a pretrained model for the CNN part. 
1. Once a basic model is there, refactor the data pipeline to use the dataset
   API. 
