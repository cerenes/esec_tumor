import json
import math
import os
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras import applications

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import keras.optimizers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools



def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)

            img = cv2.resize(img, (RESIZE, RESIZE))

            IMG.append(np.array(img))
    return IMG


normal_train = np.array(Dataset_loader('C:/Users/Ceren/Desktop/busı_3/train/normal', 224))
tumor_train = np.array(Dataset_loader('C:/Users/Ceren/Desktop/busı_3/train/tumor', 224))
normal_test = np.array(Dataset_loader('C:/Users/Ceren/Desktop/busı_3/test/normal', 224))
tumor_test = np.array(Dataset_loader('C:/Users/Ceren/Desktop/busı_3/test/tumor', 224))
normal_val = np.array(Dataset_loader('C:/Users/Ceren/Desktop/busı_3/val/normal', 224))
tumor_val = np.array(Dataset_loader('C:/Users/Ceren/Desktop/busı_3/val/tumor' ,224))


normal_train_label = np.zeros(len(normal_train))
tumor_train_label = np.ones(len(tumor_train))
normal_test_label = np.zeros(len(normal_test))
tumor_test_label = np.ones(len(tumor_test))
normal_val_label = np.zeros(len(normal_val))
tumor_val_label = np.ones(len(tumor_val))



X_train = np.concatenate((normal_train, tumor_train), axis=0)
Y_train = np.concatenate((normal_train_label,tumor_train_label), axis=0)
X_test = np.concatenate((normal_test, tumor_test), axis=0)
Y_test = np.concatenate((normal_test_label, tumor_test_label), axis=0)
X_val = np.concatenate((normal_val, tumor_val), axis=0)
Y_val = np.concatenate((normal_val_label, tumor_val_label) , axis=0)



normal_train.shape


s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

Y_train = to_categorical(Y_train, num_classes= None)
Y_test = to_categorical(Y_test, num_classes=None)

x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train,
    test_size=0.2,
    random_state=11
)



w = 60
h = 40
fig = plt.figure(figsize=(15, 15))
columns = 4
rows = 3

for i in range(1, columns * rows + 1):
    ax = fig.add_subplot(rows, columns, i)
    if np.argmax(Y_train[i]) == 0:
        ax.title.set_text('Normal')
    elif np.argmax(Y_train[i]) == 1:
        ax.title.set_text('Tümörlü')

    plt.imshow(X_train[i], interpolation='nearest')
plt.show()



BATCH_SIZE = 16

train_generator = ImageDataGenerator(
        zoom_range=2,  # rastgele yakınlaştırma
        rotation_range=90,
        horizontal_flip=True,  # görüntülerin  rastgele çevrilmesi
        vertical_flip=True,
    )


def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    # model.add(layers.Dense(128, activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.07))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    return model


resnet = tf.keras.applications.resnet.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)




model = build_model(resnet, lr=1e-4)
model.summary()

learn_control = ReduceLROnPlateau(monitor='val_accuracy', patience=5,
                                  verbose=1, factor=0.2)#, min_lr=1e-4)

filepath = "varyok3.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

retval = model.fit(
    train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=[learn_control, checkpoint]
)



plt.plot(retval.history['loss'], label = 'val_loss')
plt.plot(retval.history['accuracy'], label = 'val_accuracy')
plt.legend()
plt.grid(True)