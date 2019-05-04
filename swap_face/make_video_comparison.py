import cv2
import os

"""
    This module is to generate two videos of prediction`s frames and source frames
    
"""


def video_predictions():

    # Writes video from constructed frames
    _, _, src_files = next(os.walk('../data/predictions/'))
    file_count = len(src_files)
    img_array = []
    for i in range(file_count):

        img = cv2.imread('../data/predictions/prediction{index}.jpg'.format(index=i))
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('../data/pred_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print("Video converted.")


def video_source():

    # Writes video from constructed frames
    _, _, src_files = next(os.walk('../data/src/src_video_faces/faces/face_images'))
    file_count = len(src_files)
    img_array = []
    for i in range(file_count):

        img = cv2.imread('../data/src/src_video_faces/faces/face_images/src_face{index}.jpg'.format(index=i))
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('../data/src_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print("Video converted.")


if __name__ == "__main__":
    video_predictions()
    video_source()
