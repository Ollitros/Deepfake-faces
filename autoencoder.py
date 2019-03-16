import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input
from models import Autoencoders


def make_prediction(input_shape, path_walk, path_to_landmarks):
    combined = load_model('data/models/combined_model.h5')

    _, _, src_files = next(os.walk(path_walk))
    file_count = len(src_files)
    for i in range(file_count):
        index = src_files[i]
        index = index.split('.')
        index = index[0].split('face')
        index = int(index[1])

        src = np.asarray(cv.imread(path_to_landmarks.format(img=index)))
        src = src.astype('float32')
        src = src / 255
        src = np.reshape(src, (1, input_shape[0], input_shape[1], input_shape[2]))

        prediction = combined.predict(src)
        prediction = np.asarray(prediction)
        prediction = np.reshape(prediction[1], [input_shape[0], input_shape[1], input_shape[2]]) * 255
        cv.imwrite('data/predictions/prediction{i}.jpg'.format(i=index), prediction)


def test(X, Y):

    combined = load_model('data/models/combined_model.h5')

    print(combined.summary())

    # Test model
    # Generate Y from X
    prediction = combined.predict(X[0:2])
    for i in range(1):
        plt.subplot(231), plt.imshow(X[i], 'gray')
        plt.subplot(232), plt.imshow(Y[i], 'gray')
        plt.subplot(233), plt.imshow(prediction[0][i], 'gray')
        plt.subplot(234), plt.imshow(prediction[1][i], 'gray')
        plt.show()

    # Generate X from Y
    prediction = combined.predict(Y[0:2])
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

    # Combining two separate models into one. Required creating Input layer
    encoder_input = Input(shape=input_shape)
    encode = encoder(encoder_input)

    src_decode = src_decoder(encode)
    dst_decode = dst_decoder(encode)

    combined = Model(inputs=encoder_input, outputs=[src_decode, dst_decode])
    combined.compile(loss='mean_squared_error', optimizer='adam')
    print(combined.summary())
    combined.load_weights('data/models/combined_model.h5')

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

        prediction = combined.predict(X[0:2])
        cv.imwrite('data/temp/image{epoch}.jpg'.format(epoch=i+65), prediction[1][0]*255)

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
    train_from_video = True
    train_from_picture = False
    picture_examples = 100
    only_predict = False
    path_to_landmarks = 'data/src/src_landmark/faces/src_face{img}.jpg'
    path_walk = 'data/src/src_landmark/faces/'

    epochs = 5
    bacth_size = 2
    input_shape = (200, 200, 3)

    X = []
    Y = []
    if train_from_picture:
        # x = cv.imread('data/src/src_picture_face/src_face.jpg')
        # y = cv.imread('data/dst/dst_picture_face/dst_face.jpg')
        x = cv.imread('data/src/src_picture/src.jpg')
        y = cv.imread('data/dst/dst_picture/dst.jpg')
        x = cv.resize(x, (200, 200))
        y = cv.resize(y, (200, 200))
        for i in range(picture_examples):
            X.append(x)
            Y.append(y)
        X = np.asarray(X)
        Y = np.asarray(Y)

    elif train_from_video:
        # Count images from src folder
        _, _, src_files = next(os.walk("data/src/src_landmark/faces"))
        src_file_count = len(src_files)
        # Count images from dst folder
        _, _, dst_files = next(os.walk("data/dst/dst_landmark/faces"))
        dst_file_count = len(dst_files)
        file_count = None
        if dst_file_count > src_file_count:
            file_count = src_file_count
        elif dst_file_count < src_file_count:
            file_count = dst_file_count
        else:
            file_count = src_file_count = dst_file_count
        # Creating train dataset
        for i in range(file_count):
            image = cv.imread('data/src/src_landmark/faces/{img}'.format(img=src_files[i]))
            X.append(image)
        X = np.asarray(X)

        for i in range(file_count):
            image = cv.imread('data/dst/dst_landmark/faces/{img}'.format(img=dst_files[i]))
            Y.append(image)
        Y = np.asarray(Y)

    else:
        print("It`s fiasko, bro.")

    X = X.astype('float32')
    Y = Y.astype('float32')
    X /= 255
    Y /= 255

    if only_predict:
        make_prediction(input_shape, path_walk, path_to_landmarks)
    else:
        train(X, Y, epochs, bacth_size, input_shape)
        # test(X, Y)


if __name__ == "__main__":
    main()
