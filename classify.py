import keras
import numpy as np
import tensorflow as tf
import cv2
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing import image
from scipy.misc import imread, imresize,imshow

img_path = 'test_images/batman.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
x = np.array([img], dtype = "float32").reshape(-1, 224, 224, 3) / 255.0

keras.backend.set_image_data_format('channels_first')

model = Sequential()

model.add(Conv2D(filters = 32, input_shape = [224, 224, 3], kernel_size = 9, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 7, padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = 7, strides = 1, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 7, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 3, padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 2, padding = 'same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))

optimizer = Adam(lr = 0.0001)	

model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

model.load_weights('final.h5')

preds = model.predict_classes(x)
preds_prob = model.predict_proba(x)
print(preds_prob)

if preds == 0:
	print("Looks a lot like Batman!!")
else:
	print("Looks a lot like Joker!!")	