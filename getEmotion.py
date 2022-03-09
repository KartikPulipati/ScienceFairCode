import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    '/Users/kartik/opt/anaconda3/envs/pytorchenv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
                '/Users/kartik/opt/anaconda3/envs/pytorchenv/share/OpenCV/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(
                '/Users/kartik/opt/anaconda3/envs/pytorchenv/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')


def vConcat_Resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1]
                for img in img_list)

    # resizing images
    im_list_resize = [cv2.resize(img,
                                 (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation=interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)
#
frame0 = cv2.VideoCapture(1)
# frame1 = cv2.VideoCapture(2)


while 1:
    ret0, img1 = frame0.read()
    # ret1, img2 = frame1.read()
    img11 = cv2.resize(img1, (500, 500))
    # img22 = cv2.resize(img2, (256, 256))
    ax = []
    ay = []
    ax2 = []
    ay2 = []
    mouths = ()

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
    # if frame1:
    #     gray = cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
    #     mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=70)
    #     for (mx, my, mw, mh) in mouth:
    #         img22 = cv2.rectangle(img22, (mx, my), (mx + mw, my + mh), (0, 255, 0), 1)
    #         mouths = (mx, my, mx + mw, my + mh)
    #     cv2.imshow('Mouth', img22)

    # if len(mouths) != 0 and len(ay2) == 2:
    #     eye = img11[min(ay)-10:max(ay2)+10, min(ax)-10:max(ax2)+10]
    #     mouth = img22[mouths[1]-10:mouths[3]+10, mouths[0]-10:mouths[2]]
    #     result = vConcat_Resize([eye, mouth])
    #     cv2.imshow('Result', result)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

frame0.release()
# frame1.release()
cv2.destroyAllWindows()