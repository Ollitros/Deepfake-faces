import os
import cv2 as cv
import numpy as np


def make_dataset(resolution, splitted=None):
    X = []
    Y = []

    if splitted is not None:

        # Count images from src folder
        _, _, src_files = next(os.walk("data/training_data/src/src_video_faces/faces/face_images/output"))
        src_file_count = len(src_files)
        # Count images from dst folder
        _, _, dst_files = next(os.walk("data/training_data/dst/dst_video_faces/faces/face_images/output"))
        dst_file_count = len(dst_files)

        # Creating train dataset
        for i in range(src_file_count):
            image = cv.imread('data/training_data/src/src_video_faces/faces/face_images/output/{img}'.format(img=src_files[i]))
            image = cv.resize(image, resolution)
            X.append(image)

            if ((i + 1) % (src_file_count / splitted)) == 0:

                np.save('data/training_data/splitted/trainX{index}.npy'.format(index=i + 1), np.asarray(X))
                X = []

        for i in range(dst_file_count):
            image = cv.imread('data/training_data/dst/dst_video_faces/faces/face_images/output/{img}'.format(img=dst_files[i]))
            image = cv.resize(image, resolution)
            Y.append(image)

            if ((i + 1) % (src_file_count / splitted)) == 0:

                np.save('data/training_data/splitted/trainY{index}.npy'.format(index=i + 1), np.asarray(Y))
                Y = []

    else:

        # Count images from src folder
        _, _, src_files = next(os.walk("data/training_data/src/src_video_faces/faces/face_images/output"))
        src_file_count = len(src_files)
        # Count images from dst folder
        _, _, dst_files = next(os.walk("data/training_data/dst/dst_video_faces/faces/face_images/output"))
        dst_file_count = len(dst_files)

        # Creating train dataset
        for i in range(src_file_count):
            image = cv.imread('data/training_data/src/src_video_faces/faces/face_images/output/{img}'.format(img=src_files[i]))
            image = cv.resize(image, resolution)
            X.append(image)
        X = np.asarray(X)

        for i in range(dst_file_count):
            image = cv.imread('data/training_data/dst/dst_video_faces/faces/face_images/output/{img}'.format(img=dst_files[i]))
            image = cv.resize(image, resolution)
            Y.append(image)
        Y = np.asarray(Y)

        np.save('data/training_data/trainX.npy', X)
        np.save('data/training_data/trainY.npy', Y)


if __name__ == "__main__":

    resolution = (128, 128)
    make_dataset(resolution=resolution)