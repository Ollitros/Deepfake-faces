import cv2 as cv
import time
from super_resolution.augmentation import make_augmentation
from super_resolution.model import Gan
from keras_vggface.vggface import VGGFace
from utils import  *


def test(X, Y, input_shape):
    """
        This function does testing not by predicting directly like in train function, but
        taking prepared tensors from trained model.
        All test images will be saved in 'data/test/' folder.
    """
    model = Gan(input_shape=input_shape, image_shape=(input_shape[0], input_shape[1], 6))
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    vggface.summary()
    model.build_pl_model(vggface_model=vggface, before_activ=False)
    model.build_train_functions()
    model.load_weights()

    prediction = model.generator.predict(X[0:2])
    prediction = np.float32(prediction[0] * 255)[:, :, 1:4]
    print(prediction.shape)
    cv.imwrite('data/test.jpg', prediction)
    cv.waitKey(0)
    cv.destroyAllWindows()


def train(X, Y, epochs, batch_size, input_shape):

    # Return encoder, two decoders, and two discriminators
    model = Gan(input_shape=input_shape, image_shape=(input_shape[0], input_shape[1], 6))
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    vggface.summary()
    model.build_pl_model(vggface_model=vggface, before_activ=False)
    model.build_train_functions()

    errGA_sum = errDA_sum = 0
    display_iters = 1

    t0 = time.time()
    model.load_weights()
    iters = X.shape[0] // batch_size
    for i in range(epochs):
        # Train discriminators
        step = 0
        for iter in range(iters):
            errDA = model.train_discriminator(X=X[step: (step + batch_size)], Y=Y[step:step + batch_size])
            step = step + batch_size
            if iter % 100 == 0:
                print("Discriminator interior step", iter)

        errDA_sum += errDA[0]

        # Train generators
        step = 0
        for iter in range(iters):
            errGA = model.train_generator(X=X[step:step + batch_size], Y=Y[step:step + batch_size])
            step = step + batch_size
            if iter % 100 == 0:
                print("Generator interior step", iter)

        errGA_sum += errGA[0]

        # Visualization
        if i % display_iters == 0:
            print("----------")
            print('[iter %d] Loss_DA: %f  Loss_GA: %f  time: %f'
                  % (i, errDA_sum / display_iters,  errGA_sum / display_iters, time.time() - t0))
            print("----------")
            display_iters = display_iters + 1
        if i % 1 == 0:
            # Makes predictions after each epoch and save into temp folder.
            prediction = model.generator.predict(X[0:2])
            prediction = np.float32(prediction[0] * 255)[:, :, 1:4]
            cv.imwrite('data/models/super-resolution/temp/image{epoch}.jpg'.format(epoch=i + 90), prediction)
            model.save_weights()


def main():
    # Parameters
    epochs = 10
    batch_size = 5
    input_shape = (64, 64, 3)
    TRAIN = False

    X = np.load('data/training_data/X_sr.npy')
    Y = np.load('data/training_data/Y_sr.npy')

    X = X.astype('float32')
    Y = Y.astype('float32')

    X /= 255
    Y /= 255

    print(X.shape)
    print(Y.shape)

    if TRAIN:
        train(X, Y, epochs, batch_size, input_shape)
    else:
        test(X, Y, input_shape)


if __name__ == "__main__":
    size = 650
    shape = (64, 64)
    preprocess = False

    if preprocess:
        make_augmentation(size, shape)
    else:
        main()
