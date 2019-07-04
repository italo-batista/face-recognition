from __future__ import print_function

from os import makedirs
from os.path import exists, join

import keras
from keras.models import Model
from keras.preprocessing import image
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from math import ceil
import numpy as np
import sys
import os
import argparse, sys


global TEST_DATA_PATH, TRAIN_DATA_PATH, BATCH_SIZE, EPOCHS, IMG_DIM, num_classes

# input image dimensions
global img_dim_rows, img_dim_cols 

# https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
# https://datascience.stackexchange.com/questions/45282/generating-image-embedding-using-cnn


def get_data_generator(data_path, datagen, subset):
    generator = datagen.flow_from_directory(
        data_path,
        # The target_size is the size of your input images,every image will be resized to this size
        target_size=(img_dim_rows, img_dim_cols),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset=subset
    )

    return generator


def get_data_generators(test_data_path, train_data_path):
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        # The target_size is the size of your input images,every image will be resized to this size
        target_size=(img_dim_rows, img_dim_cols),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    train_datagen = ImageDataGenerator(
        horizontal_flip=False, validation_split=0.2)
    train_generator = get_data_generator(
        train_data_path, train_datagen, 'training')
    validation_generator = get_data_generator(
        train_data_path, train_datagen, 'validation')

    return test_generator, train_generator, validation_generator


def create_model():
        # https://keras.io/examples/tensorboard_embeddings_mnist/
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=(img_dim_rows, img_dim_cols, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name="dense_1"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name="dense_2"))

    return model


def get_num_of_classes(train_data_path, test_data_path):
    n_test_folders, n_train_folders = 0, 0
    for _, dirnames, _ in os.walk(train_data_path):
        n_train_folders += len(dirnames)
    for _, dirnames, _ in os.walk(test_data_path):
        n_test_folders += len(dirnames)
    if n_test_folders == n_train_folders:
        return n_test_folders
    else:
        raise Exception("You must have equal numers for train and test data.")


def _read_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('--test-data-path', default='validation-set')
    parser.add_argument('--train-data-path', default='aligned-images')
    parser.add_argument('--batch-size', default=128)
    parser.add_argument('--epochs', default=12)
    parser.add_argument('--img-dim', default=28)

    args=parser.parse_args()

    return args.test_data_path, args.train_data_path, int(args.batch_size), int(args.epochs), int(args.img_dim)


if __name__ == '__main__':         
    TEST_DATA_PATH, TRAIN_DATA_PATH, BATCH_SIZE, EPOCHS, IMG_DIM = _read_args()
    num_classes = get_num_of_classes(TRAIN_DATA_PATH, TEST_DATA_PATH)
    img_dim_rows = img_dim_cols = IMG_DIM
  

    test_generator, train_generator, validation_generator = get_data_generators(
        'validation-set', 'aligned-images')

    model = create_model()
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=(ceil(len(train_generator) / BATCH_SIZE)),
        epochs=EPOCHS,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=(ceil(len(validation_generator) / BATCH_SIZE))
    )


    image_path = '../data/faces/icaro.jpg'

    layer_name = 'dense_2'
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output)

    img = image.load_img(image_path, target_size=(img_dim_rows, img_dim_cols))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    intermediate_output = intermediate_layer_model.predict(x)

    print(intermediate_output[0])