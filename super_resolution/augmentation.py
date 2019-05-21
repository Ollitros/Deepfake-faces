import os
import cv2 as cv


def make_augmentation(size=600):

    # Count images from dst folder
    _, _, dst_files = next(os.walk("data/training_data/dst/dst_video_faces/faces/face_images"))

    X = []
    Y = []
    for i in range(size):

        dst = cv.imread('data/training_data/dst/dst_video_faces/faces/face_images/{img}'.format(img=dst_files[i]))
        dst_256x256 = cv.resize(dst, (256, 256))
        dst_64x64_from_32x32 = cv.resize(cv.resize(dst, (32, 32)), (64, 64))
        dst_64x64 = cv.resize(dst, (64, 64))

        X.append(dst_64x64_from_32x32)
        X.append(dst_64x64)
        Y.append(dst_256x256)
        Y.append(dst_256x256)

        # # Mirror vertically
        mv_dst_64x64_from_32x32 = cv.flip(dst_64x64_from_32x32, 0)
        mv_dst_64x64 = cv.flip(dst_64x64, 0)
        mv_dst_256x256 = cv.flip(dst_256x256, 0)

        X.append(mv_dst_64x64_from_32x32)
        X.append(mv_dst_64x64)
        Y.append(mv_dst_256x256)
        Y.append(mv_dst_256x256)

        # Mirror horizontally
        mh_dst_64x64_from_32x32 = cv.flip(dst_64x64_from_32x32, 1)
        mh_dst_64x64 = cv.flip(dst_64x64, 1)
        mh_dst_256x256 = cv.flip(dst_256x256, 1)

        X.append(mh_dst_64x64_from_32x32)
        X.append(mh_dst_64x64)
        Y.append(mh_dst_256x256)
        Y.append(mh_dst_256x256)

        # Rotate 90 Degrees Clockwise
        rc_dst_64x64_from_32x32 = cv.transpose(cv.flip(dst_64x64_from_32x32, 1))
        rc_dst_64x64 = cv.transpose(cv.flip(dst_64x64, 1))
        rc_dst_256x256 = cv.transpose(cv.flip(dst_256x256, 1))

        X.append(rc_dst_64x64_from_32x32)
        X.append(rc_dst_64x64)
        Y.append(rc_dst_256x256)
        Y.append(rc_dst_256x256)

        # Rotate 90 Degrees Counter Clockwise
        rcc_dst_64x64_from_32x32 = cv.transpose(cv.flip(dst_64x64_from_32x32, 0))
        rcc_dst_64x64 = cv.transpose(cv.flip(dst_64x64, 0))
        rcc_dst_256x256 = cv.transpose(cv.flip(dst_256x256, 0))

        X.append(rcc_dst_64x64_from_32x32)
        X.append(rcc_dst_64x64)
        Y.append(rcc_dst_256x256)
        Y.append(rcc_dst_256x256)

        # Rotate 45 degree Counter Clockwise

        rows, cols, _ = dst_64x64_from_32x32.shape
        half_dst_64x64_from_32x32 = cv.warpAffine(dst_64x64_from_32x32, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))
        rows, cols, _ = dst_64x64.shape
        half_dst_64x64 = cv.warpAffine(dst_64x64, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))
        rows, cols, _ = dst_256x256.shape
        half_dst_256x256 = cv.warpAffine(dst_256x256, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1),
                                        (cols, rows))

        X.append(half_dst_64x64_from_32x32)
        X.append(half_dst_64x64)
        Y.append(half_dst_256x256)
        Y.append(half_dst_256x256)

        # Rotate 45 degree Clockwise

        rows, cols, _ = dst_64x64_from_32x32.shape
        half_dst_64x64_from_32x32 = cv.warpAffine(dst_64x64_from_32x32,
                                                  cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
                                                  (cols, rows))
        rows, cols, _ = dst_64x64.shape
        half_dst_64x64 = cv.warpAffine(dst_64x64, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
                                       (cols, rows))
        rows, cols, _ = dst_256x256.shape
        half_dst_256x256 = cv.warpAffine(dst_256x256,
                                         cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
                                         (cols, rows))

        X.append(half_dst_64x64_from_32x32)
        X.append(half_dst_64x64)
        Y.append(half_dst_256x256)
        Y.append(half_dst_256x256)


if __name__ == "__main__":
    make_augmentation()