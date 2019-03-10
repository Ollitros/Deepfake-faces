import cv2 as cv


def video_extract(path_from, path_to):
    face_cascade = cv.CascadeClassifier('data/face_features/haarcascade_frontalface_default.xml')
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

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            # cv.imshow('frame', roi_color)
            if roi_color is not None:
                cv.imwrite(path_to.format(step=step), roi_color)
                step = step + 1

        # cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def picture_extract(path_from, path_to):
    face_cascade = cv.CascadeClassifier('data/face_features/haarcascade_frontalface_default.xml')
    image = cv.imread(path_from)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        # cv.imshow('frame', roi_color)
        if roi_color is not None:
            cv.imwrite(path_to, roi_color)

    cv.destroyAllWindows()


def main(extract_from_video, extract_from_picture):

    if extract_from_video:
        video_extract(path_from='data/src/src_video/data_src.mp4', path_to='data/src/src_video_faces/src_face{step}.jpg')
        video_extract(path_from='data/dst/dst_video/data_dst.mp4', path_to='data/dst/dst_video_faces/dst_face{step}.jpg')

    elif extract_from_picture:
        picture_extract(path_from='data/src/src_picture/src.jpg', path_to='data/src/src_picture_face/src_face.jpg')
        picture_extract(path_from='data/dst/dst_picture/dst.jpg', path_to='data/dst/dst_picture_face/dst_face.jpg')

    else:
        print("It`s error, bro.")


if __name__ == "__main__":

    extract_from_video = False
    extract_from_picture = True

    main(extract_from_video, extract_from_picture)


