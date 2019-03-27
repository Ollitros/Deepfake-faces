import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from models import Generator, Discriminator


def test(X, Y):

    encoder, src_decoder, dst_decoder = Generator((128, 128, 3))
    src_discriminator = Discriminator(image_shape=(128, 128, 3), filters=64)
    dst_discriminator = Discriminator(image_shape=(128, 128, 3), filters=64)

    # Combining two separate models into one. Required creating Input layer.
    encoder_input = Input(shape=(128, 128, 3))
    encode = encoder(encoder_input)
    src_decode = src_decoder(encode)
    dst_decode = dst_decoder(encode)
    # Create discriminators
    src_discriminate = src_discriminator(src_decode)
    dst_discriminate = dst_discriminator(dst_decode)

    combined = Model(inputs=encoder_input, outputs=[src_decode, dst_decode,
                                                    src_discriminate, dst_discriminate])
    combined.compile(loss=['mse', 'mse', 'mse', 'mse'], optimizer='adam')
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

    # Return encoder, two decoders, and two discriminators
    encoder, src_decoder, dst_decoder = Generator(input_shape)
    src_discriminator = Discriminator(image_shape=input_shape, filters=64)
    dst_discriminator = Discriminator(image_shape=input_shape, filters=64)

    # Combining two separate models into one. Required creating Input layer.
    # Create common encoder
    encoder_input = Input(shape=input_shape)
    encode = encoder(encoder_input)

    # Create generators
    src_decode = src_decoder(encode)
    dst_decode = dst_decoder(encode)

    # Create discriminators
    src_discriminate = src_discriminator(src_decode)
    dst_discriminate = dst_discriminator(dst_decode)

    combined = Model(inputs=encoder_input, outputs=[src_decode, dst_decode,
                                                    src_discriminate, dst_discriminate])
    combined.compile(loss=['mse', 'mse', 'mse', 'mse'], optimizer='adam')
    print(combined.summary())
    # combined.load_weights('data/models/combined_model.h5')

    # Adversarial ground truths
    patch = int(input_shape[0] / 2 ** 4)
    disc_patch = (patch, patch, 1)
    valid = np.ones((X.shape[0],) + disc_patch)
    fake = np.zeros((X.shape[0],) + disc_patch)

    for i in range(epochs):
        print("######################################################\n"
              "######################################################\n"
              "GLOBAL EPOCH --------------------------------------- {i}".format(i=i),
              "\n######################################################\n"
              "######################################################\n")

        encoder.trainable = True
        src_decoder.trainable = True
        dst_decoder.trainable = True

        # ################## #
        # Train discriminators
        # Get encoder output
        encoder_output_X = encoder.predict(X)
        encoder_output_Y = encoder.predict(Y)

        # Get decoders output by encoders output
        src_generated = src_decoder.predict(encoder_output_Y)
        dst_generated = dst_decoder.predict(encoder_output_X)

        # Train src discriminator
        src_discriminator.fit(X, valid, batch_size=batch_size, epochs=1)
        src_discriminator.fit(src_generated, fake, batch_size=batch_size, epochs=1)
        # Train dst discriminator
        dst_discriminator.fit(Y, valid, batch_size=batch_size, epochs=1)
        dst_discriminator.fit(dst_generated, fake, batch_size=batch_size, epochs=1)

        # ############### #
        # Train generators
        src_decoder.trainable = True
        dst_decoder.trainable = False
        src_discriminator.trainable = False
        dst_discriminator.trainable = False
        combined.compile(loss='mean_squared_error', optimizer='adam')
        combined.fit(x=X, y=[X, Y, fake, fake], epochs=1, batch_size=batch_size)

        src_decoder.trainable = False
        dst_decoder.trainable = True
        src_discriminator.trainable = False
        dst_discriminator.trainable = False
        combined.compile(loss='mean_squared_error', optimizer='adam')
        combined.fit(x=Y, y=[X, Y, fake, fake], epochs=1, batch_size=batch_size)

        # Makes predictions after each epoch and save into temp folder.
        prediction = combined.predict(X[0:2])
        cv.imwrite('data/models/temp/image{epoch}.jpg'.format(epoch=i+0), prediction[1][0]*255)
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
    train_bool = False
    test_bool = True
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
