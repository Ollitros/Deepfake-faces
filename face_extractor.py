import cv2 as cv
import numpy as np
import os
from mtcnn.mtcnn import MTCNN


def video_extract(path_from, path_to, path_to_info, path_to_frame):

    detector = MTCNN()
    cap = cv.VideoCapture(path_from)

    step = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        detection = detector.detect_faces(frame)
        if len(detection) <= 0:
            continue
        coords = detection[0]['box']

        x = coords[0]
        y = coords[1]
        w = coords[2]
        h = coords[3]
        roi_color = frame[y:y + h, x:x + w]

        if roi_color is not None:
            if w <= 100 or h <= 100:
                continue
            else:
                print(x, y, w, h)
                cv.imwrite(path_to.format(step=step), roi_color)
                cv.imwrite(path_to_frame.format(step=step), frame)
                with open(path_to_info.format(step=step), "w") as file:
                    print(x, y, w, h, file=file)
                step = step + 1

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def make_dataset():
    X = []
    Y = []

    # Count images from src folder
    _, _, src_files = next(os.walk("data/src/src_video_faces/faces/face_images"))
    src_file_count = len(src_files)
    # Count images from dst folder
    _, _, dst_files = next(os.walk("data/dst/dst_video_faces/faces/face_images"))
    dst_file_count = len(dst_files)
    file_count = None
    if dst_file_count > src_file_count:
        file_count = src_file_count
    elif dst_file_count < src_file_count:
        file_count = dst_file_count
    else:
        file_count = src_file_count = dst_file_count
    # Creating train dataset
    for i in range(file_count):
        image = cv.imread('data/src/src_video_faces/faces/face_images/{img}'.format(img=src_files[i]))
        image = cv.resize(image, (128, 128))
        X.append(image)
    X = np.asarray(X)

    for i in range(file_count):
        image = cv.imread('data/dst/dst_video_faces/faces/face_images/{img}'.format(img=dst_files[i]))
        image = cv.resize(image, (128, 128))
        Y.append(image)
    Y = np.asarray(Y)

    np.save('data/X.npy', X)
    np.save('data/Y.npy', Y)


def main(extract_from_video):

    if extract_from_video:
        video_extract(path_from='data/src/src_video/data_src.mp4', path_to='data/src/src_video_faces/faces/face_images/src_face{step}.jpg',
                      path_to_info='data/src/src_video_faces/faces/face_info/src_info{step}.txt', path_to_frame='data/src/src_video_faces/frames/src_frame{step}.jpg')
        video_extract(path_from='data/dst/dst_video/data_dst.mp4', path_to='data/dst/dst_video_faces/faces/face_images/dst_face{step}.jpg',
                      path_to_info='data/dst/dst_video_faces/faces/face_info/dst_info{step}.txt', path_to_frame='data/dst/dst_video_faces/frames/dst_frame{step}.jpg')

        make_dataset()
    else:
        print("It`s error, bro.")


if __name__ == "__main__":

    extract_from_video = True

    main(extract_from_video)



