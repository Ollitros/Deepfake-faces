import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from network.models import GanModel


def train(X, Y, epochs, batch_size, input_shape):

    # Return encoder, two decoders, and two discriminators
    model = GanModel(input_shape=input_shape, image_shape=(64, 64, 6))
    model.build_train_functions()

    errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
    display_iters = 1

    t0 = time.time()
    iters = X.shape[0] // batch_size
    model.load_weights()
    for i in range(epochs):
        print("######################################################\n"
              "GLOBAL EPOCH --------------------------------------- {i}".format(i=i),
              "\n######################################################\n")

        # Train discriminators
        step = 0
        for iter in range(iters):
            errDA, errDB = model.train_discriminators(X=X[step: (step + batch_size)], Y=Y[step:step + batch_size])
            step = step + batch_size
        errDA_sum += errDA[0]
        errDB_sum += errDB[0]

        # Train generators
        step = 0
        for iter in range(iters):
            errGA, errGB = model.train_generators(X=X[step:step + batch_size], Y=Y[step:step + batch_size])
            step = step + batch_size
        errGA_sum += errGA[0]
        errGB_sum += errGB[0]

        # Visualization
        if i % display_iters == 0:

            print("----------")
            print('[iter %d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
                  % (i, errDA_sum / display_iters, errDB_sum / display_iters,
                     errGA_sum / display_iters, errGB_sum / display_iters, time.time() - t0))
            print("----------")
            display_iters = display_iters + 1

        if i % 10 == 0:
            # Makes predictions after each epoch and save into temp folder.
            prediction = model.encoder.predict(X[0:2])
            prediction = model.dst_decoder.predict(prediction)
            prediction = np.float32(prediction[0] * 255)[:, :, 1:4]
            cv.imwrite('image{epoch}.jpg'.format(epoch=i + 0), prediction)

        model.save_weights()


def main():
    # Parameters
    epochs = 100
    batch_size = 5
    input_shape = (64, 64, 3)

    X = np.load('data/X.npy')
    Y = np.load('data/Y.npy')

    X = X.astype('float32')
    Y = Y.astype('float32')
    X /= 255
    Y /= 255

    train(X, Y, epochs, batch_size, input_shape)


if __name__ == "__main__":
    main()
