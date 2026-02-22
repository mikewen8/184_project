#https://www.tensorflow.org/tutorials/images/cnn documentation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# set the hyper parameters
data_dir = "Rice_Image_Dataset"
img_height = 224
img_width = 224
batch_size = 32
seed = 123
# documentation on tensorflow website
train_ds  = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,# trainign percentatge
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds  = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,     # same split
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

# two ways to define the model
"""
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)

model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))

# remove a layers
model.pop()
print(len(model.layers))  # 2
"""