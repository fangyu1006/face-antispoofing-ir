#input model name without h5 extension
keras_model_name = "model_allignment_irliveness_ch1_v14"
import tensorflow as tf
    


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


K.set_learning_phase(1)
top_model = Sequential()
top_model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=(112,112,1)))
top_model.add(Conv2D(64, (3, 3), activation='relu'))
top_model.add(MaxPooling2D(pool_size=(2, 2)))
top_model.add(Conv2D(128, (3, 3), activation='relu'))
top_model.add(MaxPooling2D(pool_size=(2, 2)))
top_model.add(Conv2D(64, (3, 3), activation='relu'))
top_model.add(MaxPooling2D(pool_size=(2, 2)))
#top_model.add(Dropout(0.25))
top_model.add(Flatten())
top_model.add(Dense(128, activation='relu'))
#top_model.add(Dropout(0.7))
top_model.add(Dense(3, activation='softmax'))

top_model.summary()
top_model.load_weights(keras_model_name + ".h5")




wkdir ='./frozen_models'
pb_filename =keras_model_name + ".pb"



# save model to pb ====================
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
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

# save keras model as tf pb files ===============


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in top_model.outputs])
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)
