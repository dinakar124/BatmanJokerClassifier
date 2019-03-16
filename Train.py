import cv2
import keras
import numpy as np 
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import os
from random import shuffle 
from tqdm import tqdm 
import matplotlib.pyplot as plt

train_data = 'dataset/Train'
test_data = 'dataset/Test'

keras.backend.set_image_data_format('channels_first')

def one_hot_label(img):
	label = img.split('.')[0]
	if label == 'batman':
		ohl = np.array([1,0])
	elif label == 'joker':
		ohl = np.array([0,1])
	return ohl	

def train_data_with_label():
	train_images = []
	for i in tqdm(os.listdir(train_data)):
		path = os.path.join(train_data, i)
		
		img = cv2.imread(path)
		img = cv2.resize(img, (224, 224))
		train_images.append([np.array(img), one_hot_label(i)])
	shuffle(train_images)
	return train_images

def test_data_with_label():
	test_images = []
	for i in tqdm(os.listdir(test_data)):
		path = os.path.join(test_data, i)
		img = cv2.imread(path)
		img = cv2.resize(img, (224, 224))
		test_images.append([np.array(img), one_hot_label(i)])
	return test_images	 	

training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images], dtype = "float32").reshape(-1, 224, 224, 3) / 255.0
tr_label_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images], dtype = "float32").reshape(-1, 224, 224, 3) / 255.0
tst_label_data = np.array([i[1] for i in testing_images])

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

model.fit(x = tr_img_data, y = tr_label_data, epochs = 50, batch_size = 32)

model.summary()

model.save_weights('/home/deep-bro/Downloads/Projects/Batman_Joker_classifier/final.h5')

test_loss, test_acc = model.evaluate(tst_img_data, tst_label_data, verbose = 1)

print(test_acc)


print("Done and Dusted!!")