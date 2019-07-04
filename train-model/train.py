from __future__ import print_function

from os import makedirs
from os.path import exists, join

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import numpy as np
import sys, os, argparse


global TEST_DATA_PATH, TRAIN_DATA_PATH, BATCH_SIZE, EPOCHS, IMG_DIM, num_classes

# input image dimensions
global img_dim_rows, img_dim_cols 

# https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
# https://datascience.stackexchange.com/questions/45282/generating-image-embedding-using-cnn


def get_datagen(data_path, datagen, subset):
    return datagen.flow_from_directory(
        data_path,
        target_size=(img_dim_rows, img_dim_cols),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset=subset
    )


def get_datagens(test_data_path, train_data_path):
    test_datagen = ImageDataGenerator(horizontal_flip=False)
    train_datagen = ImageDataGenerator(horizontal_flip=False, validation_split=0.2)

    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(img_dim_rows, img_dim_cols),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    train_generator = get_datagen(train_data_path, train_datagen, 'training')
    validation_generator = get_datagen(train_data_path, train_datagen, 'validation')

    return test_generator, train_generator, validation_generator


# https://keras.io/examples/tensorboard_embeddings_mnist/
def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', \
        input_shape=(img_dim_rows, img_dim_cols, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name="dense_1"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name="dense_2"))

    return model


def train(model, train_generator, validation_generator):
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(), 
        metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=(ceil(len(train_generator) / BATCH_SIZE)),
        validation_data=validation_generator,
        validation_steps=ceil(len(validation_generator) / BATCH_SIZE),
        epochs=EPOCHS,
        verbose=1,
    )
    return model


def save(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")


def _get_num_of_classes(train_data_path, test_data_path):
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

    parser.add_argument('--test-data-path', default='../data/aligned-images/test')
    parser.add_argument('--train-data-path', default='../data/aligned-images/train')
    parser.add_argument('--batch-size', default=128)
    parser.add_argument('--epochs', default=12)
    parser.add_argument('--img-dim', default=28)

    args=parser.parse_args()

    return args.test_data_path, args.train_data_path, int(args.batch_size), int(args.epochs), int(args.img_dim)


if __name__ == '__main__':         
    TEST_DATA_PATH, TRAIN_DATA_PATH, BATCH_SIZE, EPOCHS, IMG_DIM = _read_args()
    num_classes = _get_num_of_classes(TRAIN_DATA_PATH, TEST_DATA_PATH)
    img_dim_rows = img_dim_cols = IMG_DIM
  
    test_generator, train_generator, validation_generator = \
        get_datagens(TEST_DATA_PATH, TRAIN_DATA_PATH)

    model = get_model()
    train(model, train_generator, validation_generator)
    save(model)
