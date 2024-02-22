import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 200
counter = 0
name = 'Please'
folder = 'Data/'+name

font = cv2.FONT_HERSHEY_SIMPLEX
message = "Empty image crop. Skipping."
display_message = True

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgwhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset: x+w+offset]
        imgCropShape = imgCrop.shape

        if not imgCrop.size:
            continue

        aspectratio = h/w
        if aspectratio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgwhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgwhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgwhite)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f'{folder}/{name}{counter}.jpg', imgwhite)
        # imgwhite_resized = cv2.resize(imgwhite, (imgResize.shape[1], imgResize.shape[0]))
        counter += 1
        print(counter)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()