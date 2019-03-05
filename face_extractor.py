import cv2 as cv


face_cascade = cv.CascadeClassifier('data/face_features/haarcascade_frontalface_default.xml')
cap = cv.VideoCapture('data/src_video/data_src.mp4')

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
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # cv.imshow('frame', roi_color)
        cv.imwrite('data/src_faces/src_face{step}.jpg'.format(step=step), roi_color)

    # cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

    step = step + 1

cap.release()
cv.destroyAllWindows()

