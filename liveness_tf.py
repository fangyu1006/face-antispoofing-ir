

# dimensions of our images.
img_width, img_height = 112, 112
train_data_dir = './train_data/training_data_11'
validation_data_dir = './train_data/validation_data_7'
nb_train_samples = 25159#20456#45290
nb_validation_samples = 5501#5497#5214#4777#3774
epochs = 50


#==============================================================================
# TRY other network
#==============================================================================
import numpy as np
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
#from tensorflow.contrib.saved_model import save_keras_model
top_model = Sequential()
top_model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=(112,112,1)))
top_model.add(Conv2D(64, (3, 3), activation='relu'))
top_model.add(MaxPooling2D(pool_size=(2, 2)))
top_model.add(Conv2D(128, (3, 3), activation='relu'))#was 64
top_model.add(MaxPooling2D(pool_size=(2, 2)))
top_model.add(Conv2D(64, (3, 3), activation='relu'))
top_model.add(MaxPooling2D(pool_size=(2, 2)))
top_model.add(Dropout(0.25))
top_model.add(Flatten())
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.7))#was 0.5
top_model.add(Dense(3, activation='softmax'))

top_model.summary()
top_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical')

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('./model_allignment_irliveness_ch1_v14.h5', verbose=1, save_best_only=True, save_weights_only=False)
]
# fine-tune the model
top_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    callbacks=callbacks,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)















