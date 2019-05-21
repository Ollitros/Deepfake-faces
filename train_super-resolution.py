import cv2 as cv
import numpy as np
import time
from super_resolution.augmentation import make_augmentation
from network.models import GanModel
from keras_vggface.vggface import VGGFace
from utils import  *


def test(X, Y, input_shape):
    """
        This function does testing not by predicting directly like in train function, but
        taking prepared tensors from trained model.
        All test images will be saved in 'data/test/' folder.
    """
    model = GanModel(input_shape=input_shape, image_shape=(input_shape[0], input_shape[1], 6))
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    vggface.summary()
    model.build_pl_model(vggface_model=vggface, before_activ=False)
    model.build_train_functions()
    model.load_weights()

    src = X[0:2]
    dst = Y[0:2]
    eyesX = maskX[0:2]
    eyesY = maskY[0:2]

    showG_rec(src, dst, model.path_bgr_src, model.path_bgr_dst, 2)
    showG_mask(src, dst, model.path_bgr_src, model.path_bgr_dst, 2)
    showG_eyes(src, dst, eyesX, eyesY, 2)


def train(X, Y, epochs, batch_size, input_shape):

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
    iters = X.shape[0] // batch_size
    for i in range(epochs):
        # Train discriminators
        step = 0
        for iter in range(iters):
            errDA, errDB = model.train_discriminators(X=X[step: (step + batch_size)], Y=Y[step:step + batch_size])
            # model.train_dst_dis(Y=Y[step:step + batch_size])
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
            # model.train_src_gen(X=X[step:step + batch_size], maskX=maskX[step:step + batch_size])
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
    epochs = 10
    batch_size = 5
    input_shape = (64, 64, 3)
    TRAIN = True

    X = np.load('data/training_data/not_augmented_trainX.npy')
    Y = np.load('data/training_data/not_augmented_trainY.npy')

    X = X.astype('float32')
    Y = Y.astype('float32')

    X /= 255
    Y /= 255

    if TRAIN:
        train(X, Y, epochs, batch_size, input_shape)
    else:
        test(X, Y, input_shape)


if __name__ == "__main__":
    size = 650
    preprocess = True

    if preprocess:
        make_augmentation(size)
    else:
        main()
