import os
import cv2
import numpy as np
from network.models import GanModel


# Load trained model and makes face predictions
def make_prediction(input_shape, path_walk, path_to_faces):

    # Load model from 'data/models/'
    model = GanModel(input_shape=input_shape, image_shape=(64, 64, 6))
    model.load_weights(path='../data/models')

    _, _, src_files = next(os.walk(path_walk))
    file_count = len(src_files)
    for i in range(file_count):
        print(i)
        index = src_files[i]
        index = index.split('.')
        index = index[0].split('face')
        index = int(index[1])

        src = np.asarray(cv2.imread(path_to_faces.format(img=index)))
        src = src.astype('float32')
        src = src / 255
        src = cv2.resize(src, (input_shape[0], input_shape[1]))
        src = np.reshape(src, (1, input_shape[0], input_shape[1], input_shape[2]))

        prediction = model.encoder.predict(src)
        prediction = model.dst_decoder.predict(prediction)
        prediction = np.float32(prediction * 255)[:, :, :, 1:4]
        prediction = np.reshape(prediction, [input_shape[0], input_shape[1], input_shape[2]])
        cv2.imwrite('../data/predictions/prediction{i}.jpg'.format(i=index), prediction)


if __name__ == "__main__":
    path_to_faces = '../data/src/src_video_faces/faces/face_images/src_face{img}.jpg'
    path_walk = '../data/src/src_video_faces/faces/face_images/'
    input_shape = (64, 64, 3)

    make_prediction(input_shape, path_walk, path_to_faces)