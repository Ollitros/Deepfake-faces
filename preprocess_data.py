from preprocess_face.face_extractor import extract_faces
from preprocess_face.augment_data import make_augmentation
from preprocess_face.create_dataset import make_dataset
from preprocess_face.eyes_extraction import create_eyes_mask


def preprocess_data():
    # Parameters
    extract_from_video = True
    resolution = (64, 64)
    splitted = None  # 'Splitted' parameter use when dataset to huge to load in memory and you need to split it

    # extract_faces(extract_from_video=extract_from_video)
    # create_eyes_mask(resolution=resolution)
    # make_augmentation()
    make_dataset(splitted=splitted, resolution=resolution)


if __name__ == "__main__":

    preprocess_data()