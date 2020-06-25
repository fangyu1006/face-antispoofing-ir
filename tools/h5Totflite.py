import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file('./model_allignment_irliveness_ch1_mobilenet.h5',
                                                      input_arrays=['input_1'],
                                                      output_arrays=['dense_1/Softmax'])
                                                      #input_shapes={"input_1":[1,112,112,1]})

#converter = tf.lite.TocoConverter.from_keras_model_file("model_allignment_irliveness_ch1_resnet.h5")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops=True
converter.inference_type = tf.float32
converter.inference_input_type = tf.float32
tf_lite_model = converter.convert()
open('./irliveness_mobilenet.tflite', 'wb').write(tf_lite_model)
