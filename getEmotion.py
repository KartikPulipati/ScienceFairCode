import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    '/Users/kartik/opt/anaconda3/envs/pytorchenv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
                '/Users/kartik/opt/anaconda3/envs/pytorchenv/share/OpenCV/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(
                '/Users/kartik/opt/anaconda3/envs/pytorchenv/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')

frame0 = cv2.VideoCapture(0)
frame1 = cv2.VideoCapture(2)

while 1:
    ret0, img1 = frame0.read()
    ret1, img2 = frame1.read()
    img11 = cv2.resize(img1, (256, 256))
    img22 = cv2.resize(img2, (256, 256))
    ax = []
    ay = []
    ax2 = []
    ay2 = []
    mouths = 0

    if frame0:
        gray = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            img11 = cv2.rectangle(img11, (ex, ey), (ex+ew, ey+eh), (0, 256, 0), 1)
            ax.append(ex)
            ay.append(ey)
            ax2.append(ex+ew)
            ay2.append(ey+eh)

        cv2.imshow('Eye', img11)
    if frame1:
        gray = cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
        mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=70)
        for (mx, my, mw, mh) in mouth:
            img22 = cv2.rectangle(img22, (mx, my), (mx + mw, my + mh), (0, 256, 0), 1)
            mouths = (mx, my, mx + mw, my + mh)
        cv2.imshow('Mouth', img22)

        if ax and ay and ax2 and ay2 and mouth != 0:
            result = np.concatenate(img11[min(ax):max(ax2), min(ay):max(ay2)], img22[mouths[0]:mouths[2], mouths[1]:mouths[2]])
            cv2.imshow('Result', result)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
      break

frame0.release()
frame1.release()
cv2.destroyAllWindows()