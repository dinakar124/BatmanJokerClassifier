# BatmanJokerClassifier
This repository includes an image classification model which classifies Batman and joker. I've included the code to train the model(Train.py) and also the code for classifying your own image(classify.py). 

I've personally collected the dataset from Google images using Batchkun Image Downloader extension.

Download the dataset from here : https://drive.google.com/open?id=11dieEypwGy93cv7qHRSFl2P9YKue1XjH

Required Dependencies : Python, Tensorflow, Keras

Tensorflow and Keras Installation : https://www.tensorflow.org/install/pip (Tensorflow)
                                    https://keras.io/#installation (Keras)

First, clone the repository and download the dataset.
Change the dataset path to the path where you downloaded the dataset in Train.py

Running "Train.py" trains the model and produces the keras h5 weights file (which basically contains the saved weights of the model).

"final.h5" is the file which contains the weights of our model.

In "Classify.py", we deploy our trained Keras model to production so that we can test it on our own images.
