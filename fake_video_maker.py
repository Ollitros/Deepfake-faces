import cv2
import os


_, _, src_files = next(os.walk('data/src/src_video_faces/faces/face_images/'))
file_count = len(src_files)
img_array = []
for i in range(file_count):

    img = cv2.imread('data/src/src_video_faces/faces/face_images/src_face{index}.jpg'.format(index=i))
    img = cv2.resize(img, (200, 200))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


out = cv2.VideoWriter('data/deep_fake_video/deep_fake.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()