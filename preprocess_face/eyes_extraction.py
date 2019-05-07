import face_alignment
import cv2
import numpy as np
from glob import glob
from pathlib import PurePath, Path
from matplotlib import pyplot as plt


def create_eyes_mask(resolution):
    # Code from https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    fns_face_not_detected = []

    dir_faceA = "data/training_data/src/src_video_faces/faces/face_images"
    dir_faceB = "data/training_data/dst/dst_video_faces/faces/face_images"
    dir_bm_faceA_eyes = "data/training_data/binary_masks/face_src_eyes"
    dir_bm_faceB_eyes = "data/training_data/binary_masks/face_dst_eyes"

    fns_faceA = glob(f"{dir_faceA}/*.*")
    fns_faceB = glob(f"{dir_faceB}/*.*")

    # !mkdir -p binary_masks/faceA_eyes
    Path(f"data/training_data/binary_masks/face_src_eyes").mkdir(parents=True, exist_ok=True)
    # !mkdir -p binary_masks/faceB_eyes
    Path(f"data/training_data/binary_masks/face_dst_eyes").mkdir(parents=True, exist_ok=True)

    for idx, fns in enumerate([fns_faceA, fns_faceB]):
        if idx == 0:
            save_path = dir_bm_faceA_eyes
        elif idx == 1:
            save_path = dir_bm_faceB_eyes

            # create binary mask for each training image
        for fn in fns:
            raw_fn = PurePath(fn).parts[-1]

            x = plt.imread(fn)
            preds = fa.get_landmarks(x)

            if preds is not None:
                preds = preds[0]
                mask = np.zeros_like(x)

                # Draw right eye binary mask
                pnts_right = [(preds[i, 0], preds[i, 1]) for i in range(36, 42)]
                hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
                mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

                # Draw left eye binary mask
                pnts_left = [(preds[i, 0], preds[i, 1]) for i in range(42, 48)]
                hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
                mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

                # Draw mouth binary mask
                # pnts_mouth = [(preds[i,0],preds[i,1]) for i in range(48,60)]
                # hull = cv2.convexHull(np.array(pnts_mouth)).astype(np.int32)
                # mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)

                mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
                mask = cv2.GaussianBlur(mask, (7, 7), 0)

            else:
                mask = np.zeros_like(x)
                print(f"No faces were detected in image '{fn}''")
                fns_face_not_detected.append(fn)

            plt.imsave(fname=f"{save_path}/{raw_fn}", arr=mask, format="jpg")
            print(fn)


if __name__ == "__main__":

    resolution = (64, 64)

    create_eyes_mask(resolution=resolution)