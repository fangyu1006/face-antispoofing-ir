import numpy as np
import keras
from keras.models import model_from_json
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow import Tensor

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
from resnet import resnet
from mobilenetv2 import MobileNetv2


def build_model(net):
    if net is "mobilenet":
        model = MobileNetv2((112,112,1), 3, 0.25)
    elif net is "resnet":
        model = resnet()
    return model




if __name__ == "__main__":
    # dimensions of our images.
    img_width, img_height = 112, 112
    train_data_dir = './train_data/training_data_12'
    validation_data_dir = './train_data/validation_data_6'
    nb_train_samples = 15984#20456#45290
    nb_validation_samples = 13167#5497#5214#4777#3774
    epochs = 50

    model = build_model("mobilenet")
    model.summary
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



