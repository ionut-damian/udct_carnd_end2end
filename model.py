from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Lambda
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import ntpath
import numpy as np
import cv2
import matplotlib.pyplot as plt

batch_size = 64
epochs = 5

# input image dimensions
input_shape = (160, 320, 3)

def prepare_data(path):
    lines = []
    with open(path + '/' + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)     
    return lines
            
def generator(path, samples, batch_size=32):
    n_samples = len(samples)
    while(1):
        shuffle(samples)
        #iterate through all batches
        for i in range(0, n_samples, batch_size):
            # grab batch
            batch_samples = samples[i:i+batch_size]
            # load data from HDD
            images = []
            measurements = []
            for sample in batch_samples:
                # iterate through all images of a sample
                for j in range(3):
                    # load image
                    source_path = sample[j]
                    filename = ntpath.basename(source_path)
                    current_path = path + '/IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    # load steering angle
                    meas = float(sample[3])
                    # apply correction factor depending on which image this is
                    correction_factor = 0.2
                    if i == 1: # left camera
                        meas += correction_factor
                    elif i == 2: # right camera
                        meas -= correction_factor
                    measurements.append(meas)
                    # generate adidtional data by flipping each image
                    images.append(cv2.flip(image, 1))
                    measurements.append(meas * -1.0)   

            # yield results
            x_train = np.array(images, dtype='float32')
            y_train = np.array(measurements, dtype='float32')
            yield shuffle(x_train, y_train)
                    
def create_model():
    model = Sequential()
    # preproc
    model.add(Cropping2D(cropping=((65,22), (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # convolutions
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # flatten
    model.add(Dropout(0.25))
    model.add(Flatten())
    # dense
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    # output
    model.add(Dense(1))    
    return model

################################
# MAIN
path = '../data'
samples = prepare_data(path)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# create generators for data
train_generator = generator(path, train_samples, batch_size=batch_size)
validation_generator = generator(path, validation_samples, batch_size=batch_size)

print("train samples: " + str(len(train_samples)))
print("valid samples: " + str(len(validation_samples)))

model = create_model()
model.compile(loss='mse',
              optimizer='adam')

history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
          validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,
          epochs=epochs,
          verbose=1)

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

model.save('model.h5')