import cv2
import os


def main():

    # Writes video from constructed frames
    _, _, src_files = next(os.walk('../data/swapped_frames/'))
    file_count = len(src_files)
    img_array = []
    for i in range(file_count):

        img = cv2.imread('../data/swapped_frames/swapped_frame{index}.jpg'.format(index=i))
        if img is None:
            continue
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('../data/deep_fake_video/deep_fake.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print("Video converted.")


if __name__ == "__main__":
    main()