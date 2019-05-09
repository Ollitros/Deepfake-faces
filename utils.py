import numpy as np
import cv2

"""
    Code from https://github.com/shaoanlu/faceswap-GAN/blob/master/utils.py    
"""


def get_transpose_axes(n):
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
    return y_axes, x_axes, [n - 1]


def stack_images(images):
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
    ).reshape(new_shape)


def showG_rec(test_A, test_B, path_A, path_B, batchSize):
    figure_A = np.stack([
        test_A,
        np.squeeze(np.array([path_A([test_A[i:i + 1]]) for i in range(test_A.shape[0])])),
        np.squeeze(np.array([path_B([test_A[i:i + 1]]) for i in range(test_A.shape[0])])),
    ], axis=1)
    figure_B = np.stack([
        test_B,
        np.squeeze(np.array([path_B([test_B[i:i + 1]]) for i in range(test_B.shape[0])])),
        np.squeeze(np.array([path_A([test_B[i:i + 1]]) for i in range(test_B.shape[0])])),
    ], axis=1)

    cv2.imwrite('data/test/src_to_dst.jpg', ((figure_A[0][2] + 1) * 255).astype('uint8'))
    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, batchSize // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    cv2.imwrite('data/test/reconstructed.jpg', figure)


def showG_mask(test_A, test_B, path_A, path_B, batchSize):
    figure_A = np.stack([
        test_A,
        (np.squeeze(np.array([path_A([test_A[i:i + 1]]) for i in range(test_A.shape[0])]))) * 2 - 1,
        (np.squeeze(np.array([path_B([test_A[i:i + 1]]) for i in range(test_A.shape[0])]))) * 2 - 1,
    ], axis=1)
    figure_B = np.stack([
        test_B,
        (np.squeeze(np.array([path_B([test_B[i:i + 1]]) for i in range(test_B.shape[0])]))) * 2 - 1,
        (np.squeeze(np.array([path_A([test_B[i:i + 1]]) for i in range(test_B.shape[0])]))) * 2 - 1,
    ], axis=1)

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, batchSize // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    cv2.imwrite('data/test/masked.jpg', figure)


def showG_eyes(test_A, test_B, bm_eyes_A, bm_eyes_B, batchSize):
    figure_A = np.stack([
        (test_A + 1) / 2,
        bm_eyes_A,
        bm_eyes_A * (test_A + 1) / 2,
    ], axis=1)
    figure_B = np.stack([
        (test_B + 1) / 2,
        bm_eyes_B,
        bm_eyes_B * (test_B + 1) / 2,
    ], axis=1)

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, batchSize // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip(figure * 255, 0, 255).astype('uint8')
    cv2.imwrite('data/test/eyes.jpg', figure)