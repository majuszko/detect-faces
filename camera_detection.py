import cv2 as cv

cap = cv.VideoCapture(0)

def detection():
    global cap

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

    cv.namedWindow('frame')
    cv.namedWindow('follow')

    k = ord(' ')

    while k != ord('q'):
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)

        imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detect_faces = face_cascade.detectMultiScale(imgray, 1.7, 4)
        #detect_eye = eye_cascade.detectMultiScale(imgray, 2.4, 4)
        face = 0
        for (x, y, w, h) in detect_faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
            roi_gray = imgray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 3, 1)
            cv.imshow('follow', roi_color)
            cv.putText(frame, f'face {face}', (x, y), cv.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 1)
            face += 1
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv.putText(frame, f'Total people : {face}', (40, 70), cv.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 2)

        cv.imshow('frame', frame)


        k = cv.waitKey(10)

    cap.release()
    cv.destroyAllWindows()

detection()