import cv2 as cv


def extract_faces(path_from, path_to):
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


extract_faces(path_from='data/src_video/data_src.mp4', path_to='data/src_faces/src_face{step}.jpg')
extract_faces(path_from='data/dst_video/data_dst.mp4', path_to='data/dst_faces/dst_face{step}.jpg')
