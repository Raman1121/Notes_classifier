from keras.models import Sequential
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.activations import relu
from keras.layers import Conv2D

model = Sequential()

#First Layer
model.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

#Second Layer
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

#Adding the Flattening
model.add(Flatten())

#Adding two Fully Connected Layers
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

##model.fit_generator(training_set,
                         ##steps_per_epoch = 8000,
                         ##epochs = 25,
                         ##validation_data = test_set,
                         ##validation_steps = 2000)

#Importing the image for testing 
import numpy as np 
from keras.preprocessing import image

test_image = image.load_img('path', target_size= (64,64))  #Path to the camera folder in the gallery

#Converting image to numpy array
test_image = image.img_to_array(test_image)

#Adding another Dimention 
test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)