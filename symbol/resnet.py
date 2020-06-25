import numpy as np
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, ReLU, BatchNormalization,Add
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow import Tensor
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


def resnet():
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
    return model

if __name__ == "__main__":
    model = create_resnet()
    model.summary()


