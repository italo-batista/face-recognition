from keras.models import model_from_json
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import argparse, sys, os


global img_dim_rows, img_dim_cols


def load_model(model_json_file_path='model.json', weights_file_path='model.h5'):
    json_file = open(model_json_file_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_file_path)

    return loaded_model


def _read_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--img-dim', default=28)
    args=parser.parse_args()

    return int(args.img_dim)


def load_imgs_paths():
    col_dir = "../data/aligned-images/test/"
    imgs = []
    for full_path, dirnames, filenames in os.walk(col_dir):
        if len(dirnames) == 0:  # there is no subdir in current dir
            full_filenames_paths = map(
                lambda x: full_path + "/" + x,
                filenames
            )
            imgs = imgs + [img for img in full_filenames_paths]
    return imgs


def get_img_embeddings(img_path):
    img = image.load_img(image_path, target_size=(img_dim_rows, img_dim_cols))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(img)
    intermediate_output = intermediate_layer_model.predict(x)
    embeddings = intermediate_output[0]
    return embeddings


def is_header(line, sep):
    line_splitted = line.split(sep)
    items_types = [type(item) for item in line_splitted]
    only_str = all(item_type == str for item_type in items_types)
    return only_str


if __name__ == '__main__':         
    IMG_DIM = _read_args()
    img_dim_rows = img_dim_cols = IMG_DIM

    model = load_model()
    last_layer_name = 'dense_2'
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(last_layer_name).output)

    filepath = "tolerance_exp.csv"
    first_time = os.path.isfile(filepath)
    mode = "w" if first_time else "a"

    imgs = load_imgs_paths()
    with open(filepath, mode) as fp:
        if not first_time:
            fp.truncate(0)
        
        fp.write("%s;%s\n" % ("label", ";".join(["x"+str(n) for n in range(250)])))

        for image_path in imgs:
            img_class = image_path.split("/")[-2]
            embeddings = get_img_embeddings(image_path)
            
            embeddings_str = list(map(str, embeddings)) 
            embeddings_inline = ";".join(embeddings_str)

            fp.write("%s;%s\n" % (img_class, embeddings_inline))





            