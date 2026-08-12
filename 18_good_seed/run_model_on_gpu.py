
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


def load_train(path, labels):

    """
    Carga la parte de entrenamiento del conjunto de datos desde la ruta.
    """

    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1/255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        width_shift_range=.20,
        height_shift_range=.20
    )

    train_gen_flow = train_datagen.flow_from_dataframe(
        labels,
        directory=path,
        x_col='file_name',
        y_col='real_age',
        class_mode='raw',
        target_size=(224, 224),
        batch_size=32,
        subset='training',
        seed=42
    )

    return train_gen_flow


def load_test(path, labels):

    """
    Carga la parte de validación/prueba del conjunto de datos desde la ruta
    """

    test_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1/255
    )

    test_gen_flow = test_datagen.flow_from_dataframe(
        labels,
        directory=path,
        x_col='file_name',
        y_col='real_age',
        class_mode='raw',
        target_size=(224, 224),
        batch_size=32,
        subset='validation',
        seed=42
    )

    return test_gen_flow


def create_model(input_shape):

    """
    Define el modelo
    """

    backbone = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    model = Sequential()

    model.add(backbone)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=1, activation='relu'))

    optimizer = Adam(learning_rate=0.0001)

    model.compile(
        loss='mae',
        optimizer=optimizer,
        metrics=['acc']
    )

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20):

    """
    Entrena el modelo dados los parámetros
    """

    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    return model


