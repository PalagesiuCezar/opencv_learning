import cv2
import numpy as np

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 52, 72])
    upper_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    low_red = np.array([94, 80, 2])
    high_red = np.array([126, 255, 255])
    red_mask = cv2.inRange(hsv, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    cv2.imshow("frame", frame)
    cv2.imshow("green_mask", green)
    cv2.imshow("red_mask", red)
    key = cv2.waitKey(25)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
