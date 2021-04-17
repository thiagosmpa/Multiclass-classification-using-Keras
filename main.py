'''
    Multiclass classifier using Keras.
    If the model is already saved, comment everything and load it to save
    time and processing.
    
'''
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
import keras
from tensorflow.keras import layers
from dataset_generator import createDataBase as dg
from matplotlib import pyplot

# =============================================================================
# Preparing data from the dataset_generator
# =============================================================================
# This line will run the another code "dataset_generator". Run this if you
# don't have the prepared data to continue the task. 

dg.dataPrepare()




dataset_read = pd.read_csv('dataset.csv', delimiter=',')
Img = dataset_read.loc[0,:]
image_read = cv2.imread('dataset/' + str(Img[0]) + '/' + str(Img[1]))
image = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)

image_size = image.shape
batch_size = 32

# =============================================================================
# Model Creator
# =============================================================================

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)


    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)


    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes


    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)





# =============================================================================
# Training e validating sets
# =============================================================================

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


# =============================================================================
# Data augmentation. In this case, flipping and rotating the images
# =============================================================================

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)


# =============================================================================
# Creating and fitting the Model
# =============================================================================

model = make_model(input_shape=image_size + (3,), num_classes=42)
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

epochs = 1
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,)



# =============================================================================
# Saving Model for future evaluating
# =============================================================================

# if (os.path.isdir('Model') == False):
#     os.mkdir('Model')
# model.save('Model/')


# =============================================================================
# Test
# =============================================================================

# model = keras.models.load_model('Model/')
# If Model is already saved, load here to save time fitting again


img = keras.preprocessing.image.load_img(
    "dataset/0/0_5.jpg", target_size=image_size
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
# 'predictions' give the probability to each class
