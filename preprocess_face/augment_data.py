import Augmentor
import os


def make_augmentation(samples):
    # Make source augmentation
    folder = '../data/src/src_resized/output'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    p = Augmentor.Pipeline('../data/src/src_resized/')
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    p.rotate_random_90(probability=0.5)
    p.skew_corner(probability=0.5, magnitude=0.2)
    p.random_distortion(probability=0.5, magnitude=3, grid_height=2, grid_width=2)
    p.shear(probability=0.5, max_shear_left=5, max_shear_right=5)
    p.flip_random(probability=0.5)
    p.sample(samples)

    # Make destination augmentation
    folder = '../data/dst/dst_resized/output'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    p = Augmentor.Pipeline('../data/dst/dst_resized/')
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    p.rotate_random_90(probability=0.5)
    p.skew_corner(probability=0.5, magnitude=0.2)
    p.random_distortion(probability=0.5, magnitude=3, grid_height=2, grid_width=2)
    p.shear(probability=0.5, max_shear_left=5, max_shear_right=5)
    p.flip_random(probability=0.5)
    p.sample(samples)


if __name__ == "__main__":
    make_augmentation(samples=1000)