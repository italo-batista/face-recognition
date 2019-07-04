from keras.models import model_from_json
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import argparse, sys

def load_model(model_json_file_path='model.json', weights_file_path='model.h5'):
    json_file = open(model_json_file_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_file_path)

    return loaded_model

if __name__ == '__main__': 
    model = load_model()

    image_path = '../data/faces/icaro.jpg'

    layer_name = 'dense_2'
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output)

    img = image.load_img(image_path, target_size=(28, 28))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    intermediate_output = intermediate_layer_model.predict(x)

    print(intermediate_output[0])

