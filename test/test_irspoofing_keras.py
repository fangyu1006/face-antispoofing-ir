import cv2
import tensorflow as tf 
import numpy as np
import os
import keras
from keras import backend as K

import sys
import os
sys.path.append("../symbol"))
from resnet import resnet
from mobilenetv2 import MobileNetv2 


def load_model(weight_path, net):
    K.set_learning_phase(0)
    if net is "mobilenet":
        top_model = MobileNetv2((112,112,1), 3, 0.25) 
    elif net is "resnet":
        top_model = resnet()
    top_model.load_weights(weight_path)
    return top_model 


def extract_feature_batch(images, model):
    images = images.astype(float)
    images = images*0.0078125
    embeddings = model.predict(images)
    return embeddings

if __name__ == "__main__":
    src_dir = "../test_data"
    model_path = "../models/model_allignment_irliveness_ch1_mobilenet.h5"
    model = load_model(model_path, "mobilenet")
    wrong = 0
    total = 0

    print("test on " + src_dir)
    for root, dirs, files in os.walk(src_dir):
        for name in files:
            img_path = os.path.join(root, name)
            images_patch = []
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img).reshape(112,112,1)
            images_patch.append(img)
            feature = extract_feature_batch(np.asarray(images_patch), model)
            total += 1
            if (feature[0][0] > 0.95):
                wrong += 1
                print(feature)

    print("total num: " + str(total))
    print("wrong num: " + str(wrong))
