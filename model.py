from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Lambda
from keras.layers import Cropping2D

import csv
import ntpath
import numpy as np
import cv2

batch_size = 64
epochs = 10

# input image dimensions
input_shape = (160, 320, 3)

def load_data(path):
    lines = []
    with open(path + '/' + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = ntpath.basename(source_path)
            current_path = path + '/IMG/' + filename
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

            meas = float(line[3])
            correction_factor = 0.2
            if i == 1: # left camera
                meas += correction_factor
            elif i == 2: # right camera
                meas -= correction_factor
            measurements.append(meas)
            
            # flip image
            images.append(cv2.flip(image, 1))
            measurements.append(meas * -1.0)

    return np.array(images, dtype='float32'), np.array(measurements, dtype='float32')

x_train, y_train  = load_data('./data')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

model = Sequential()
# preproc
model.add(Cropping2D(cropping=((65,22), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# convolutions
#model.add(Conv2D(6, kernel_size=(5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(6, kernel_size=(5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

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

model.compile(loss='mse',
              optimizer='adam')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2,
          shuffle=True)

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

model.save('model.h5')