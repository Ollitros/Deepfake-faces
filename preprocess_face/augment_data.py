import os
import cv2 as cv


def make_augmentation():

    src_augmented_path = 'data/training_data/src/src_augmented/'
    dst_augmented_path = 'data/training_data/dst/dst_augmented/'
    mask_src_augmented_path = 'data/training_data/binary_masks/mask_src_augmented/'
    mask_dst_augmented_path = 'data/training_data/binary_masks/mask_dst_augmented/'

    # Count images from src folder
    _, _, src_files = next(os.walk("data/training_data/src/src_video_faces/faces/face_images"))
    src_file_count = len(src_files)
    # Count images from dst folder
    _, _, dst_files = next(os.walk("data/training_data/dst/dst_video_faces/faces/face_images"))
    dst_file_count = len(dst_files)

    # Count images from binary src folder
    _, _, mask_src_files = next(os.walk("data/training_data/binary_masks/face_src_eyes"))
    mask_src_file_count = len(mask_src_files)
    # Count images from binary dst folder
    _, _, mask_dst_files = next(os.walk("data/training_data/binary_masks/face_dst_eyes"))
    mask_dst_file_count = len(mask_dst_files)

    file_count = 0
    if mask_dst_file_count >= mask_src_file_count:
        file_count = mask_src_file_count
    else:
        file_count = mask_dst_file_count

    count = 0
    for i in range(file_count):
        src = cv.imread('data/training_data/src/src_video_faces/faces/face_images/{img}'.format(img=src_files[i]))
        dst = cv.imread('data/training_data/dst/dst_video_faces/faces/face_images/{img}'.format(img=dst_files[i]))
        mask_src = cv.imread('data/training_data/binary_masks/face_src_eyes/{img}'.format(img=mask_src_files[i]))
        mask_dst = cv.imread('data/training_data/binary_masks/face_dst_eyes/{img}'.format(img=mask_dst_files[i]))

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), mask_dst)

        # Mirror vertically
        count = count + 1
        mv_src = cv.flip(src, 0)
        mv_dst = cv.flip(dst, 0)
        mv_mask_src = cv.flip(mask_src, 0)
        mv_mask_dst = cv.flip(mask_dst, 0)

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), mv_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), mv_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), mv_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), mv_mask_dst)

        # Mirror horizontally
        count = count + 1
        mh_src = cv.flip(src, 1)
        mh_dst = cv.flip(dst, 1)
        mh_mask_src = cv.flip(mask_src, 1)
        mh_mask_dst = cv.flip(mask_dst, 1)

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), mh_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), mh_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), mh_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), mh_mask_dst)

        # Rotate 90 Degrees Clockwise
        count = count + 1
        rc_src = cv.transpose(cv.flip(src, 1))
        rc_dst = cv.transpose(cv.flip(dst, 1))
        rc_mask_src = cv.transpose(cv.flip(mask_src, 1))
        rc_mask_dst = cv.transpose(cv.flip(mask_dst, 1))

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), rc_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), rc_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), rc_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), rc_mask_dst)

        # Rotate 90 Degrees Counter Clockwise
        count = count + 1
        rcc_src = cv.transpose(cv.flip(src, 0))
        rcc_dst = cv.transpose(cv.flip(dst, 0))
        rcc_mask_src = cv.transpose(cv.flip(mask_src, 0))
        rcc_mask_dst = cv.transpose(cv.flip(mask_dst, 0))

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), rcc_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), rcc_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), rcc_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), rcc_mask_dst)

        # Rotate 180 Degrees (Same as Flipping vertically and horizontally at the same time)
        count = count + 1
        rr_src = cv.flip(src, -1)
        rr_dst = cv.flip(dst, -1)
        rr_mask_src = cv.flip(mask_src, -1)
        rr_mask_dst = cv.flip(mask_dst, -1)

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), rr_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), rr_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), rr_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), rr_mask_dst)

        # Rotate 180 Degrees (Same as Flipping vertically and horizontally at the same time) + transpose
        count = count + 1
        rrt_src = cv.transpose(cv.flip(src, -1))
        rrt_dst = cv.transpose(cv.flip(dst, -1))
        rrt_mask_src = cv.transpose(cv.flip(mask_src, -1))
        rrt_mask_dst = cv.transpose(cv.flip(mask_dst, -1))

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), rrt_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), rrt_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), rrt_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), rrt_mask_dst)

        # Rotate 45 degree Counter Clockwise
        count = count + 1
        rows, cols, _ = src.shape
        half_src = cv.warpAffine(src, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))
        rows, cols, _ = dst.shape
        half_dst = cv.warpAffine(dst, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))
        rows, cols, _ = mask_src.shape
        half_mask_src = cv.warpAffine(mask_src, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))
        rows, cols, _ = mask_dst.shape
        half_mask_dst = cv.warpAffine(mask_dst, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), half_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), half_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), half_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), half_mask_dst)

        # Rotate 45 degree Clockwise
        count = count + 1
        rows, cols, _ = src.shape
        half_src = cv.warpAffine(src, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1), (cols, rows))
        rows, cols, _ = dst.shape
        half_dst = cv.warpAffine(dst, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1), (cols, rows))
        rows, cols, _ = mask_src.shape
        half_mask_src = cv.warpAffine(mask_src, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
                                      (cols, rows))
        rows, cols, _ = mask_dst.shape
        half_mask_dst = cv.warpAffine(mask_dst, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
                                      (cols, rows))

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), half_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), half_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), half_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), half_mask_dst)

        # Rotate 45 degree Counter Clockwise from flipped vertically
        count = count + 1
        rows, cols, _ = src.shape
        half_src = cv.warpAffine(mv_src, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))
        rows, cols, _ = dst.shape
        half_dst = cv.warpAffine(mv_dst, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1), (cols, rows))
        rows, cols, _ = mask_src.shape
        half_mask_src = cv.warpAffine(mv_mask_src, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1),
                                      (cols, rows))
        rows, cols, _ = mask_dst.shape
        half_mask_dst = cv.warpAffine(mv_mask_dst, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 1),
                                      (cols, rows))

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), half_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), half_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), half_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), half_mask_dst)

        # Rotate 45 degree Clockwise from flipped vertically
        count = count + 1
        rows, cols, _ = src.shape
        half_src = cv.warpAffine(mv_src, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
                                 (cols, rows))
        rows, cols, _ = dst.shape
        half_dst = cv.warpAffine(mv_dst, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
                                 (cols, rows))
        rows, cols, _ = mask_src.shape
        half_mask_src = cv.warpAffine(mv_mask_src, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
                                      (cols, rows))
        rows, cols, _ = mask_dst.shape
        half_mask_dst = cv.warpAffine(mv_mask_dst, cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1),
                                      (cols, rows))

        cv.imwrite(src_augmented_path + 'src{i}.jpg'.format(i=count), half_src)
        cv.imwrite(dst_augmented_path + 'dst{i}.jpg'.format(i=count), half_dst)
        cv.imwrite(mask_src_augmented_path + 'mask_src{i}.jpg'.format(i=count), half_mask_src)
        cv.imwrite(mask_dst_augmented_path + 'mask_dst{i}.jpg'.format(i=count), half_mask_dst)

        count = count + 1


if __name__ == "__main__":
    make_augmentation()