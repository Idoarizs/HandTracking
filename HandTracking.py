import cv2
import mediapipe as mp

camera = cv2.VideoCapture(cv2.CAP_DSHOW)
mpHands = mp.solutions.hands
hand = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, imghdr = camera.read()
    imgRGB = cv2.cvtColor(imghdr, cv2.COLOR_BGRA2RGB)
    result = hand.process(imgRGB)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(imghdr, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = imghdr.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)

    cv2.imshow("Camera", imghdr)
    cv2.waitKey(1)