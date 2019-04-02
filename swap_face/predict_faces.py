import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Load trained model and makes face predictions
def make_prediction(input_shape, path_walk, path_to_faces):
    combined = load_model('../data/models/combined_model.h5')

    _, _, src_files = next(os.walk(path_walk))
    file_count = len(src_files)
    for i in range(file_count):
        index = src_files[i]
        index = index.split('.')
        index = index[0].split('face')
        index = int(index[1])

        src = np.asarray(cv2.imread(path_to_faces.format(img=index)))
        src = src.astype('float32')
        src = src / 255
        src = np.reshape(src, (1, input_shape[0], input_shape[1], input_shape[2]))

        prediction = combined.predict(src)
        prediction = np.asarray(prediction)
        # prediction = cv2.fastNlMeansDenoisingColored(prediction, None, 10, 10, 7, 21)
        prediction = np.reshape(prediction[1], [input_shape[0], input_shape[1], input_shape[2]]) * 255
        cv2.imwrite('../data/predictions/prediction{i}.jpg'.format(i=index), prediction)


if __name__ == "__main__":
    path_to_faces = '../data/src/src_video_faces/faces/src_face{img}.jpg'
    path_walk = '../data/src/src_video_faces/faces/'
    input_shape = (200, 200, 3)

    make_prediction(input_shape, path_walk, path_to_faces)