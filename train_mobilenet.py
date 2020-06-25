

#==============================================================================
# TRY other network
#==============================================================================
import numpy as np
import keras
from keras.models import model_from_json
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, ReLU, BatchNormalization,Add
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow import Tensor

from mobilenetv2 import MobileNetv2
#from tensorflow.contrib.saved_model import save_keras_model

def relu_bn(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x, downsample=False, filters=64, kernel_size=3):
    y = Conv2D(kernel_size=kernel_size,
	       strides = (1 if not downsample else 2),
               filters = filters,
               padding = 'same')(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides = 1,
               filters = filters,
               padding = 'same')(y)
    if downsample:
        x = Conv2D(kernel_size = 1,
                   strides = 2,
                   filters = filters,
                   padding = 'same')(x)

    out = Add()([x, y])
    out = relu_bn(out)
    return out


def create_mobilenet():
    model = MobileNetv2((112,112,1), 3, 0.25)
    model.summary()
    return model



def create_resnet():
    inputs = Input(shape=(112, 112, 1))
    num_filters = 32

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size = 3,
               strides = 1,
               filters = num_filters,
               padding = 'same')(t)
    t = relu_bn(t)
    
    num_blocks_list = [1,1]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2

    t = MaxPooling2D(pool_size=(3, 3))(t)
    t = Flatten()(t)
    #t = Dense(128, activation='relu')(t)
    outputs = Dense(3, activation='softmax')(t)
    
    model = Model(inputs, outputs)
    model.summary()
    return model

if __name__ == "__main__":
    # dimensions of our images.
    img_width, img_height = 112, 112
    train_data_dir = './train_data/training_data_12'
    validation_data_dir = './train_data/validation_data_6'
    nb_train_samples = 15984#20456#45290
    nb_validation_samples = 13167#5497#5214#4777#3774
    epochs = 50

    model = create_mobilenet()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
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
        ModelCheckpoint('./model_allignment_irliveness_ch1_mobilenet.h5', verbose=1, save_best_only=True, save_weights_only=False)
    ]
    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        callbacks=callbacks,
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)



