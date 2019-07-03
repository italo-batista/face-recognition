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
import sys
import numpy as np

batch_size = 128
num_classes = 22  # TODO
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
# https://datascience.stackexchange.com/questions/45282/generating-image-embedding-using-cnn


def get_data_generator(data_path, datagen, subset):
    generator = datagen.flow_from_directory(
        data_path,
        # The target_size is the size of your input images,every image will be resized to this size
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset
    )

    return generator


def get_data_generators(test_data_path, train_data_path):
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        # The target_size is the size of your input images,every image will be resized to this size
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
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
                     input_shape=(img_rows, img_cols, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name="dense_1"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name="dense_2"))

    return model


test_generator, train_generator, validation_generator = get_data_generators(
    'validation-set', 'aligned-images')

model = create_model()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(ceil(len(train_generator) / batch_size)),
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=(ceil(len(validation_generator) / batch_size))
)

try:
    image_path = sys.argv[1]
except:
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
