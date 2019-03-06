import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from models import Autoencoder


def test():
    input_shape = (200, 200, 3)

    ##################
    # Load src model
    src_model = Autoencoder(input_shape)
    src_model.load_weights('data/models/model_src.h5')

    # Test model
    prediction = src_model.predict(X[0:10])

    for i in range(5):
        plt.subplot(231), plt.imshow(prediction[i], 'gray')
        plt.subplot(232), plt.imshow(X[i], 'gray')
        plt.show()

    ##################
    # Load dst model
    dst_model = Autoencoder(input_shape)
    dst_model.load_weights('data/models/model_dst.h5')

    # Test model
    prediction = dst_model.predict(Y[0:10])

    for i in range(5):
        plt.subplot(231), plt.imshow(prediction[i], 'gray')
        plt.subplot(232), plt.imshow(Y[i], 'gray')
        plt.show()


def train():
    input_shape = (200, 200, 3)

    ##################
    # Create src model
    src_model = Autoencoder(input_shape)

    # checkpoint
    filepath = "data/models/src/src-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    print(src_model.summary())

    src_model.fit(X, X, epochs=200, batch_size=25, validation_data=(X, X), callbacks=[checkpoint])
    src_model.save('data/models/model_src.h5')

    # Test model
    prediction = src_model.predict(X[0:10])

    for i in range(3):
        plt.subplot(231), plt.imshow(prediction[i], 'gray')
        plt.subplot(232), plt.imshow(X[i], 'gray')
        plt.show()

    ##################
    # Create dst model
    dst_model = Autoencoder(input_shape)

    # checkpoint
    filepath = "data/models/dst/dst-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    print(dst_model.summary())

    dst_model.fit(Y, Y, epochs=200, batch_size=25, validation_data=(Y, Y), callbacks=[checkpoint])
    dst_model.save('data/models/model_dst.h5')

    # Test model
    prediction = dst_model.predict(Y[0:10])

    for i in range(3):
        plt.subplot(231), plt.imshow(prediction[i], 'gray')
        plt.subplot(232), plt.imshow(Y[i], 'gray')
        plt.show()


# Count images from src folder
_, _, src_files = next(os.walk("data/src_faces"))
src_file_count = len(src_files)

# Count images from dst folder
_, _, dst_files = next(os.walk("data/dst_faces"))
dst_file_count = len(dst_files)

file_count = None
if dst_file_count > src_file_count:
    file_count = src_file_count
elif dst_file_count < src_file_count:
    file_count = dst_file_count
else:
    file_count = src_file_count = dst_file_count


# Creating train dataset
X = []
for i in range(file_count):
    image = cv.imread('data/src_faces/{img}'.format(img=src_files[i]))
    image = cv.resize(image, (200, 200))
    X.append(image)
X = np.asarray(X)

Y = []
for i in range(file_count):
    image = cv.imread('data/dst_faces/{img}'.format(img=dst_files[i]))
    image = cv.resize(image, (200, 200))
    Y.append(image)
Y = np.asarray(Y)

# Normalize dataset before training
X = X.astype('float32')
Y = Y.astype('float32')
X /= 255
Y /= 255

# train()
test()
