import os
import cv2 as cv
import numpy as np


def make_dataset():
    X = []
    Y = []

    # Count images from src folder
    _, _, src_files = next(os.walk("../data/src/src_resized/output"))
    src_file_count = len(src_files)
    # Count images from dst folder
    _, _, dst_files = next(os.walk("../data/dst/dst_resized/output"))
    dst_file_count = len(dst_files)

    # Creating train dataset
    for i in range(src_file_count):
        image = cv.imread('../data/src/src_resized/output/{img}'.format(img=src_files[i]))
        X.append(image)
    X = np.asarray(X)

    for i in range(dst_file_count):
        image = cv.imread('../data/dst/dst_resized/output/{img}'.format(img=dst_files[i]))
        Y.append(image)
    Y = np.asarray(Y)

    np.save('../data/training_data/trainX.npy', X)
    np.save('../data/training_data/trainY.npy', Y)


if __name__ == "__main__":

    make_dataset()