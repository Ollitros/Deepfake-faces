import os
import cv2 as cv
import numpy as np


def make_augmentation(size=600, shape=(128, 128)):

    # Count images from dst folder
    _, _, dst_files = next(os.walk("data/training_data/dst/dst_video_faces/faces/face_images"))

    X = []
    Y = []
    for i in range(size):

        dst = cv.imread('data/training_data/dst/dst_video_faces/faces/face_images/{img}'.format(img=dst_files[i]))
        dst_256x256 = cv.resize(dst, shape)
        dst_64x64_from_32x32 = cv.resize(cv.resize(dst, (16, 16)), shape)
        dst_64x64 = cv.resize(cv.resize(dst, (32, 32)), shape)

        X.append(cv.resize(dst_64x64_from_32x32, shape))
        X.append(cv.resize(dst_64x64, shape))
        Y.append(dst_256x256)
        Y.append(dst_256x256)

        # # # Mirror vertically
        # mv_dst_64x64_from_32x32 = cv.flip(dst_64x64_from_32x32, 0)
        # mv_dst_64x64 = cv.flip(dst_64x64, 0)
        # mv_dst_256x256 = cv.flip(dst_256x256, 0)
        #
        # X.append(cv.resize(mv_dst_64x64_from_32x32, shape))
        # X.append(cv.resize(mv_dst_64x64, shape))
        # Y.append(mv_dst_256x256)
        # Y.append(mv_dst_256x256)

        # Mirror horizontally
        mh_dst_64x64_from_32x32 = cv.flip(dst_64x64_from_32x32, 1)
        mh_dst_64x64 = cv.flip(dst_64x64, 1)
        mh_dst_256x256 = cv.flip(dst_256x256, 1)

        X.append(cv.resize(mh_dst_64x64_from_32x32, shape))
        X.append(cv.resize(mh_dst_64x64, shape))
        Y.append(mh_dst_256x256)
        Y.append(mh_dst_256x256)

        # Rotate 90 Degrees Clockwise
        rc_dst_64x64_from_32x32 = cv.transpose(cv.flip(dst_64x64_from_32x32, 1))
        rc_dst_64x64 = cv.transpose(cv.flip(dst_64x64, 1))
        rc_dst_256x256 = cv.transpose(cv.flip(dst_256x256, 1))

        X.append(cv.resize(rc_dst_64x64_from_32x32, shape))
        X.append(cv.resize(rc_dst_64x64, shape))
        Y.append(rc_dst_256x256)
        Y.append(rc_dst_256x256)

        # Rotate 90 Degrees Counter Clockwise
        rcc_dst_64x64_from_32x32 = cv.transpose(cv.flip(dst_64x64_from_32x32, 0))
        rcc_dst_64x64 = cv.transpose(cv.flip(dst_64x64, 0))
        rcc_dst_256x256 = cv.transpose(cv.flip(dst_256x256, 0))

        X.append(cv.resize(rcc_dst_64x64_from_32x32, shape))
        X.append(cv.resize(rcc_dst_64x64, shape))
        Y.append(rcc_dst_256x256)
        Y.append(rcc_dst_256x256)

        # # Rotate 45 degree Counter Clockwise
        #
        # rows, cols, _ = dst_64x64_from_32x32.shape
        # half_dst_64x64_from_32x32 = cv.warpAffine(dst_64x64_from_32x32, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))
        # rows, cols, _ = dst_64x64.shape
        # half_dst_64x64 = cv.warpAffine(dst_64x64, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))
        # rows, cols, _ = dst_256x256.shape
        # half_dst_256x256 = cv.warpAffine(dst_256x256, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1),
        #                                 (cols, rows))
        #
        # X.append(cv.resize(half_dst_64x64_from_32x32, shape))
        # X.append(cv.resize(half_dst_64x64, shape))
        # Y.append(half_dst_256x256)
        # Y.append(half_dst_256x256)
        #
        # # Rotate 45 degree Clockwise
        #
        # rows, cols, _ = dst_64x64_from_32x32.shape
        # half_dst_64x64_from_32x32 = cv.warpAffine(dst_64x64_from_32x32,
        #                                           cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
        #                                           (cols, rows))
        # rows, cols, _ = dst_64x64.shape
        # half_dst_64x64 = cv.warpAffine(dst_64x64, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
        #                                (cols, rows))
        # rows, cols, _ = dst_256x256.shape
        # half_dst_256x256 = cv.warpAffine(dst_256x256,
        #                                  cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
        #                                  (cols, rows))
        #
        # X.append(cv.resize(half_dst_64x64_from_32x32, shape))
        # X.append(cv.resize(half_dst_64x64, shape))
        # Y.append(half_dst_256x256)
        # Y.append(half_dst_256x256)

    np.save('data/training_data/X_sr.npy', X)
    np.save('data/training_data/Y_sr.npy', Y)


if __name__ == "__main__":
    make_augmentation()