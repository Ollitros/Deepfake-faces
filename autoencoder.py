import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from models import Autoencoders


def test():

    # Create autoencoder with two branches
    input_shape = (200, 200, 3)
    encoder, src_decoder, dst_decoder = Autoencoders(input_shape)

    # Combining two separate models into one. Required creating Input layer
    encoder_input = Input(shape=(200, 200, 3))
    encode = encoder(encoder_input)

    src_decode = src_decoder(encode)
    dst_decode = dst_decoder(encode)

    combined = Model(inputs=encoder_input, outputs=[src_decode, dst_decode])
    combined.compile(loss='mean_squared_error', optimizer='adam')
    combined.load_weights('data/models/combined_model.h5')
    print(combined.summary())

    # Test model
    # Generate Y from X
    prediction = combined.predict(X[0:10])
    for i in range(3):
        plt.subplot(231), plt.imshow(X[i], 'gray')
        plt.subplot(232), plt.imshow(Y[i], 'gray')
        plt.subplot(233), plt.imshow(prediction[0][i], 'gray')
        plt.subplot(234), plt.imshow(prediction[1][i], 'gray')
        plt.show()

    # Generate X from Y
    prediction = combined.predict(Y[0:10])
    for i in range(3):
        plt.subplot(231), plt.imshow(X[i], 'gray')
        plt.subplot(232), plt.imshow(Y[i], 'gray')
        plt.subplot(233), plt.imshow(prediction[0][i], 'gray')
        plt.subplot(234), plt.imshow(prediction[1][i], 'gray')
        plt.show()


def train(epochs, batch_size):

    # Return encoder and two decoders
    input_shape = (200, 200, 3)
    encoder, src_decoder, dst_decoder = Autoencoders(input_shape)

    # Create checkpoint
    filepath = "data/models/checkpoints/combined-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    # Combining two separate models into one. Required creating Input layer
    encoder_input = Input(shape=(200, 200, 3))
    encode = encoder(encoder_input)

    src_decode = src_decoder(encode)
    dst_decode = dst_decoder(encode)

    combined = Model(inputs=encoder_input, outputs=[src_decode, dst_decode])
    combined.compile(loss='mean_squared_error', optimizer='adam')
    print(combined.summary())

    for i in range(epochs):
        print("######################################################\n"
              "######################################################\n"
              "GLOBAL EPOCH --------------------------------------- {i}".format(i=i),
              "\n######################################################\n"
              "######################################################\n")

        src_decoder.trainable = True
        dst_decoder.trainable = False
        combined.compile(loss='mean_squared_error', optimizer='adam')
        combined.fit(x=X, y=[X, Y], epochs=2, batch_size=batch_size, callbacks=[checkpoint], validation_data=(X, [X, Y]))

        src_decoder.trainable = False
        dst_decoder.trainable = True
        combined.compile(loss='mean_squared_error', optimizer='adam')
        combined.fit(x=Y, y=[X, Y], epochs=2, batch_size=batch_size, callbacks=[checkpoint], validation_data=(Y, [X, Y]))

    combined.save('data/models/combined_model.h5')

    # Test model
    prediction = combined.predict(X[0:10])

    for i in range(1):
       plt.subplot(231), plt.imshow(X[i], 'gray')
       plt.subplot(232), plt.imshow(Y[i], 'gray')
       plt.subplot(233), plt.imshow(prediction[0][i], 'gray')
       plt.subplot(234), plt.imshow(prediction[1][i], 'gray')
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

epochs = 10
bacth_size = 25

# train(epochs, bacth_size)
test()
