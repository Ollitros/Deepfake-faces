import cv2 as cv
import numpy as np
import time
from network.models import GanModel
from keras_vggface.vggface import VGGFace


def train(X, Y, maskX, maskY, epochs, batch_size, input_shape, splitted):

    # Return encoder, two decoders, and two discriminators
    model = GanModel(input_shape=input_shape, image_shape=(input_shape[0], input_shape[1], 6))
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    vggface.summary()
    model.build_pl_model(vggface_model=vggface, before_activ=False)
    model.build_train_functions()

    errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
    display_iters = 1

    t0 = time.time()
    model.load_weights()
    if splitted is not None:

        for i in range(epochs):

            for index in range(splitted):
                X = np.load('data/training_data/splitted/trainX{index}.npy'.format(index=(index + 1)*1000))
                Y = np.load('data/training_data/splitted/trainY{index}.npy'.format(index=(index + 1)*1000))
                X = X.astype('float32')
                Y = Y.astype('float32')
                X /= 255
                Y /= 255

                iters = X.shape[0] // batch_size
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
            print("----------")
            print('[iter %d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
                  % (i, errDA_sum / display_iters, errDB_sum / display_iters,
                     errGA_sum / display_iters, errGB_sum / display_iters, time.time() - t0))
            print("----------")
            display_iters = display_iters + 1

            # Makes predictions after each epoch and save into temp folder.
            prediction = model.encoder.predict(X[0:2])
            prediction = model.dst_decoder.predict(prediction)
            prediction = np.float32(prediction[0] * 255)[:, :, 1:4]
            cv.imwrite('image{epoch}.jpg'.format(epoch=i + 0), prediction)
            model.save_weights()

    else:
        iters = X.shape[0] // batch_size
        for i in range(epochs):

            # Train discriminators
            step = 0
            for iter in range(iters):
                errDA, errDB = model.train_discriminators(X=X[step: (step + batch_size)], Y=Y[step:step + batch_size])
                step = step + batch_size
                if iter % 100 == 0:
                    print("Discriminator interior step", iter)
            errDA_sum += errDA[0]
            errDB_sum += errDB[0]

            # Train generators
            step = 0
            for iter in range(iters):
                errGA, errGB = model.train_generators(X=X[step:step + batch_size], Y=Y[step:step + batch_size],
                                                      maskX=maskX[step:step + batch_size], maskY=maskY[step:step + batch_size])
                step = step + batch_size
                if iter % 100 == 0:
                    print("Generator interior step", iter)
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

            if i % 1 == 0:
                # Makes predictions after each epoch and save into temp folder.
                prediction = model.encoder.predict(X[0:2])
                prediction = model.dst_decoder.predict(prediction)
                prediction = np.float32(prediction[0] * 255)[:, :, 1:4]
                cv.imwrite('data/models/temp/image{epoch}.jpg'.format(epoch=i + 0), prediction)

                model.save_weights()


def main():
    # Parameters
    epochs = 7
    batch_size = 5
    input_shape = (64, 64, 3)
    splitted = None  # 'Splitted' parameter use when dataset is huge to load in memory and you need to split it

    if splitted is not None:

        train(X=None, Y=None, maskX=0, maskY=0, epochs=epochs, batch_size=batch_size, input_shape=input_shape, splitted=splitted)

    else:
        X = np.load('data/training_data/trainX.npy')
        Y = np.load('data/training_data/trainY.npy')
        maskX = np.load('data/training_data/maskX.npy')
        maskY = np.load('data/training_data/maskY.npy')

        X = X.astype('float32')
        Y = Y.astype('float32')
        maskX = maskX.astype('float32')
        maskY = maskY.astype('float32')
        X /= 255
        Y /= 255
        maskX /= 255
        maskY /= 255

        train(X, Y, maskX, maskY, epochs, batch_size, input_shape, splitted)


if __name__ == "__main__":
    main()
