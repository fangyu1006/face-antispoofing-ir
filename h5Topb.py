import tensorflow as tf

import numpy as np 
import keras
from keras.models import model_from_json
from keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
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


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


if __name__ == "__main__":
    #input model name without h5 extension
    keras_model_name = "model_allignment_irliveness_ch1_mobilenet"
    K.set_learning_phase(0)

    top_model = build_model("mobilenet")
    top_model.load_weights("models/" + keras_model_name + ".h5")
    wkdir ='./frozen_models'
    pb_filename = keras_model_name + ".pb"


    frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in top_model.outputs])
    tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)
