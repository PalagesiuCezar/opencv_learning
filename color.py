import cv2
import numpy as np

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

while(1):
        _, img = cap.read()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        low_blue = np.array([94, 80, 2])
        high_blue = np.array([126, 255, 255])

        blue_mask = cv2.inRange(hsv, low_blue, high_blue)
        kernal = np.ones((5 ,5), "uint8")

        # blue=cv2.dilate(yellow, kernal)

        res=cv2.bitwise_and(img, img, mask = blue_mask)

        (contours,hierarchy)=cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for frame, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                # approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
                # x = approx.ravel()[0]
                # y = approx.ravel()[1]
                if(area > 2000): #300
                    x, y, w, h = cv2.boundingRect(contour)
                    # print("W: {} \t H: {} \t X: {} Y: {}".format(w, h, x, y))
                    rectangle = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(img, "Rectangle", (x, y), font, 0.75, (255,0,0), 1)
                    x_center = int((x+(x+w))/2)
                    y_center = int((y+(y+h))/2)
                    # print("X_C: {} \t Y_C: {} \n".format(x_center, y_center))
                    center = cv2.rectangle(img, (x_center, y_center), (x_center+1, y_center+1), (255, 255, 0), 5)


        cv2.imshow("Color Tracking", img)
        img = cv2.flip(img,1)
        cv2.imshow("Blue",res)

        if cv2.waitKey(10) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
