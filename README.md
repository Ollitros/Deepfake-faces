# Deepfake-faces
My independent implementation of Deepfake. 
This is only **pet-project**.
# How to use
1) Put source video named **'data_src.mp4'** into **'data/src/src_video/'** folder.
2) Put destination video named **'data_dst.mp4'** into **'data/dst/dst_video/'** folder.
3) Use **preprocess_data.py** for extracting faces, metadata, frames and make data augmentation from **"data/src/src_video/data_src.mp4"** 
folder into **"data/src/src_video_faces/faces/face_images/"** ,  **"data/src/src_video_faces/faces/face_info/"** and
**"data/src/src_video_faces/frames/"** directories. 
This module makes the same operation for dst folder.
4) Train autoencoder from **train_autoencoder.py** module. **Epochs** and **batch_size** you can change inside **main** function.
There are all model files in **'data/models/'** folder.
5) Use **predict_faces.py** to generate faces from train model. Generated images you can find in **'data/predictions/'** directory.
6) Use **face_swap.py** to take original frames, faces`s metadata and predicted images to swap and write in **'data/swapped_frames/'** folder.
7) Use **fake_video_maker.py** to create final fake video and save it in **'data/deep_fake_video/'** folder.
# Recommendations
1) The **Deepfake-faces** pet-project works well only with videos where faces can be ease recognized. 
2) This code works only with one face in video. **REMEMBER THIS**. 
3) Batch_size should be not less then 4 and no more then 32.
4) Amount of global epochs should be 10 000 - 15 000.