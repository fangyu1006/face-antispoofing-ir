import cv2
import tensorflow as tf 
import numpy as np
import os


def get_model(model_str):
    interpreter = tf.lite.Interpreter(model_path=model_str)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def run_model(images, interpreter, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], images.astype('float32'))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data



def extract_feature_batch(images, interpreter, input_details, output_details):
    images = images.astype(float)
    images = images*0.0078125
    embeddings = run_model(images, interpreter, input_details, output_details)
    return embeddings

if __name__ == "__main__":
    src_dir = "/home/fangyu/Pictures/rgb-ir/no_mask/"
    model_path = "./model/irliveness_res5.tflite"
    interpreter, input_details, output_details = get_model(model_path)
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
            feature = extract_feature_batch(np.asarray(images_patch), interpreter, input_details, output_details)
            total += 1
            if (feature[0][0] > 0.95):
                wrong += 1

    print("total num: " + str(total))
    print("wrong num: " + str(wrong))
