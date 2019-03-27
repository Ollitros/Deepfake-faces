import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input
from models import Autoencoders
import models


def test(X, Y):

    encoder, src_decoder, dst_decoder = models.Autoencoders((128, 128, 3))
    # Combining two separate models into one. Required creating Input layer.
    encoder_input = Input(shape=(128, 128, 3))
    encode = encoder(encoder_input)
    src_decode = src_decoder(encode)
    dst_decode = dst_decoder(encode)
    combined = Model(inputs=encoder_input, outputs=[src_decode, dst_decode])
    combined.compile(loss='mean_squared_error', optimizer='adam')
    print(combined.summary())
    combined.load_weights('data/combined_model.h5')

    prediction = combined.predict(X[0:2])
    for i in range(1):
        plt.subplot(231), plt.imshow(X[i], 'gray')
        plt.subplot(232), plt.imshow(Y[i], 'gray')
        plt.subplot(233), plt.imshow(prediction[0][i], 'gray')
        plt.subplot(234), plt.imshow(prediction[1][i], 'gray')
        plt.show()


def train(X, Y, epochs, batch_size, input_shape):

    # Return encoder and two decoders
    encoder, src_decoder, dst_decoder = Autoencoders(input_shape)

    # Create checkpoint
    filepath = "data/models/checkpoints/combined-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    # Combining two separate models into one. Required creating Input layer.
    encoder_input = Input(shape=input_shape)
    encode = encoder(encoder_input)

    src_decode = src_decoder(encode)
    dst_decode = dst_decoder(encode)

    combined = Model(inputs=encoder_input, outputs=[src_decode, dst_decode])
    combined.compile(loss='mean_squared_error', optimizer='adam')
    print(combined.summary())
    # combined.load_weights('data/models/combined_model.h5')

    for i in range(epochs):
        print("######################################################\n"
              "######################################################\n"
              "GLOBAL EPOCH --------------------------------------- {i}".format(i=i),
              "\n######################################################\n"
              "######################################################\n")

        src_decoder.trainable = True
        dst_decoder.trainable = False
        combined.compile(loss='mean_squared_error', optimizer='adam')
        combined.fit(x=X, y=[X, Y], epochs=1, batch_size=batch_size, callbacks=[checkpoint], validation_data=(X, [X, Y]))

        src_decoder.trainable = False
        dst_decoder.trainable = True
        combined.compile(loss='mean_squared_error', optimizer='adam')
        combined.fit(x=Y, y=[X, Y], epochs=1, batch_size=batch_size, callbacks=[checkpoint], validation_data=(Y, [X, Y]))

        # Makes predictions after each epoch and save into temp folder.
        prediction = combined.predict(X[0:2])
        cv.imwrite('data/models/temp/image{epoch}.jpg'.format(epoch=i+200), prediction[1][0]*255)
        combined.save('data/models/combined_model.h5')

    combined.save('data/models/combined_model.h5')

    # Test model
    prediction = combined.predict(X[0:2])

    for i in range(1):
       plt.subplot(231), plt.imshow(X[i], 'gray')
       plt.subplot(232), plt.imshow(Y[i], 'gray')
       plt.subplot(233), plt.imshow(prediction[0][i], 'gray')
       plt.subplot(234), plt.imshow(prediction[1][i], 'gray')
       plt.show()


def main():
    # Parameters
    train_bool = True
    test_bool = False
    epochs = 5
    batch_size = 10
    input_shape = (128, 128, 3)

    X = np.load('data/X.npy')
    Y = np.load('data/Y.npy')

    X = X.astype('float32')
    Y = Y.astype('float32')
    X /= 255
    Y /= 255

    if train_bool:
        train(X, Y, epochs, batch_size, input_shape)
    elif test_bool:
        test(X, Y)
    else:
        print("It`s fiasko, bro.")


if __name__ == "__main__":
    main()
