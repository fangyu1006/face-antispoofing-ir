import cv2
import tensorflow as tf 
import numpy as np
import os




def load_graph_tf(frozen_graph_filename, name_input_node,name_output_node):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name='')

    input_tensor  = graph.get_tensor_by_name(name_input_node)
    output_tensor = graph.get_tensor_by_name(name_output_node)
    net = tf.Session(graph=graph)
    return net,input_tensor,output_tensor


def run_model(images,net,input_tensor,output_tensor):
    feed_dict  = {input_tensor:images}
    return net.run(output_tensor, feed_dict)


def extract_feature_batch(images, net, input_tensor, output_tensor):
    images = images.astype(float)
    images = images*0.0078125
    embeddings = run_model(images,net,input_tensor,output_tensor)
    return embeddings

if __name__ == "__main__":
    src_dir = "/home/fangyu/Pictures/rgb-ir/mask/"
    model_path = "./model/model_allignment_irliveness_ch1_mobilenet.pb"
    net, inputs, prelogits = load_graph_tf(model_path,"input_1:0","reshape_2/Reshape:0")
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
            feature = extract_feature_batch(np.asarray(images_patch), net, inputs, prelogits)
            total += 1
            if (feature[0][0] > 0.95):
                wrong += 1

    print("total num: " + str(total))
    print("wrong num: " + str(wrong))
