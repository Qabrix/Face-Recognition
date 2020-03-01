from os import path

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications

from DataGenerator import DataGenerator
from ConstValues import (
    PROJECT_DIR,
    DATASET_DIR_NAME,
    LABELS_FILE_NAME,
    TRAINING_LABELS_DIR_NAME,
    VALIDATION_LABELS_DIR_NAME,
    TRAINING_DIRS,
    VALIDATION_DIRS,
)

BATCH_SIZE = 32
IMG_HEIGHT, IMG_WIDTH = 120, 160

def train_model(model, train_datagen, val_datagen, epochs):
    history = model.fit_generator(
    generator =  train_datagen,
    validation_data = val_datagen,
    epochs=epochs,
    steps_per_epoch = train_datagen.__len__(),
    validation_steps = val_datagen.__len__(),
    verbose=1)

    model.save('model.h5')

def unfreeze_layers(model):
    for layer in model.layers:
        layer.trainable = True

def create_model():
    input_tensor = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
    base_model = applications.ResNet50(weights='imagenet',include_top= False,input_tensor=input_tensor)
    for layer in base_model.layers:
        layer.trainable = False

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(2))

    model = Model(inputs = base_model.input, output= top_model(base_model.output))
    #model.load_weights('model_saved.h5')
    return model

def main():
    model = create_model()
    model.summary()

    train_datagen = DataGenerator(
        directory = path.join(PROJECT_DIR, DATASET_DIR_NAME), 
        target_json =  path.join(PROJECT_DIR, TRAINING_LABELS_DIR_NAME, LABELS_FILE_NAME), 
        batch_size = BATCH_SIZE,
        target_size = (IMG_HEIGHT,IMG_WIDTH),
        classes_to_consider = TRAINING_DIRS,
        translate = 0.1,
        scale_min = 0.9,
        scale_max = 1.1,
        shear = 0.3,
        rotation = 45,
        flip = True,
        shuffle = True)

    val_datagen = DataGenerator(
        directory = path.join(PROJECT_DIR, DATASET_DIR_NAME), 
        target_json =  path.join(PROJECT_DIR, VALIDATION_LABELS_DIR_NAME, LABELS_FILE_NAME), 
        batch_size = BATCH_SIZE,
        target_size = (IMG_HEIGHT,IMG_WIDTH),
        classes_to_consider = VALIDATION_DIRS,
        translate = 0.15,
        scale_min = 0.9,
        scale_max = 1.1,
        shear = 0.3,
        flip = True,
        rotation = 30,
        shuffle = True)

    opt = Adam(lr = 0.010)
    model.compile(loss = 'mse', optimizer = opt, metrics = ['mae'])

    train_model(model, train_datagen, val_datagen, epochs=4)

    opt = Adam(lr = 0.001)
    model.compile(loss = 'mse', optimizer = opt, metrics = ['mae'])
    train_model(model, train_datagen, val_datagen, epochs=4)

    unfreeze_layers(model)
    model.compile(loss = 'mse', optimizer = opt, metrics = ['mae'])
    train_model(model, train_datagen, val_datagen, epochs=10)

    train_datagen = DataGenerator(
        directory = path.join(PROJECT_DIR, DATASET_DIR_NAME), 
        target_json =  path.join(PROJECT_DIR, TRAINING_LABELS_DIR_NAME, LABELS_FILE_NAME), 
        batch_size = BATCH_SIZE,
        target_size = (IMG_HEIGHT,IMG_WIDTH),
        classes_to_consider = TRAINING_DIRS,
        translate = 0.2,
        scale_min = 0.8,
        scale_max = 1.3,
        shear = 0.5,
        rotation = 70,
        flip = True,
        shuffle = True)

    model.compile(loss = 'mean_squared_error', optimizer = 'adagrad', metrics = ['mae'])
    train_model(model, train_datagen, val_datagen, epochs=20)

main()