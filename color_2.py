import cv2
import numpy as np
import time
import math
import sys
#Capturing Video through webcam.
def atapulii(centru_x1,centru_x2,centru_y1,centru_y2): #pitag
        ipot = math.sqrt(((centru_x1-centru_x2)**2)+((centru_y1-centru_y2)**2))
        return ipot
def pula_muie_cur(centers,k):
                boase = atapulii(centers[k-1][0],centers[k][0],centers[k-1][1],centers[k][1])
                print(boase)
                # print(len(centers))
                # if (len(centers)
                if int(boase) < 200:
                        try:
                                pula_muie_cur(centers, k+1)
                        except:
                                print("nu exista nici un")
                else: 
                        print("a gasit pathul")
                        # sys.exit(1)
                # sum_x = centers[k][0] + centers[k+1][0]
                        # sum_y = centers[k][1] + centers[k+1][1]
                        # print("SUM_X: {}".format(sum_x))
                        # print("SUM_Y: {}".format(sum_y))

def main():
        while(1):
                fps, img = cap.read()

                #converting frame(img) from BGR (Blue-Green-Red) to HSV (hue-saturation-value)

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                #defining the range of blue color
                low_blue = np.array([94, 55, 50]) #94, 80, 2 #94, 50, 50
                high_blue = np.array([130, 255, 255]) #126, 255, 255

                low_green = np.array([50, 50, 50])
                high_green = np.array([102, 255, 255])


                #finding the range blue colour in the image
                blue_mask = cv2.inRange(hsv, low_blue, high_blue)
                green_mask = cv2.inRange(hsv, low_green, high_green)
                # mask = cv2.inRange(hsv, low, upper)

                #Morphological transformation, Dilation
                kernal = np.ones((5 ,5), "uint8")

                # blue=cv2.dilate(yellow, kernal)


                res_blue=cv2.bitwise_and(img, img, mask = blue_mask)
                res_green=cv2.bitwise_and(img, img, mask = green_mask)

                # Tracking Colour (Blue)
                (contours,hierarchy)=cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                rectangle = []
                centers = []
                sum_x = 0
                sum_y = 0

                # centers = [()]
                for frame, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        # approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
                        # x = approx.ravel()[0]
                        # y = approx.ravel()[1]
                        x_center = 0
                        y_center = 0

                        if(area > 2000): #300
                                x, y, w, h = cv2.boundingRect(contour)
                                border = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
                                rectangle.append(border)
                                for rectangles in range(0, len(rectangle)):
                                        cv2.putText(img, "Rectangle", (x, y), font, 0.75, (255,0,0), 1)
                                        x_center = int((2*x+w)/2)
                                        y_center = int((2*y+h)/2)
                                        # for k in range (0, len(centers) - 1):
                                        #     sum_x += centers[rectangles][0]
                                        # for m in range (0, len(centers) - 1):
                                        #     sum_y += centers[rectangles][1]

                                        # print("X1: {} X2: {}".format(centers[0], centers[2]))
                                        #print("W{}: {} \t H{}: {}".format(rectangles, w, rectangles, h))
                                        #print("X_C{}: {} \t Y_C{}: {} \n".format(rectangles, x_center, rectangles, y_center))
                                        # for k in range (0, len(centers) - 1):
                                        #     diff_x -= centers[k][0]
                                        #     diff_y -= centers[k][1]
                                        #     if k >= stop:
                                        #         break
                                        #     print("SUM_X: {}".format(diff_x))
                                        #     print("SUM_Y: {}".format(diff_y))
                                        # time.sleep(0.5)
                                        center_tuple = (x_center, y_center)
                                centers.append(center_tuple)
                                if len(centers) >= 2:        
                                        for k in range(0, len(centers) - 1):
                                        
                                                print("XC_1: {} XC_2: {}".format(centers[k-1][0], centers[k][0]))
                                                print("YC_1: {} YC_2: {}".format(centers[k-1][1], centers[k][1]))
                                        
                                                pula_muie_cur(centers,k)                     
                                #print("X_C{}: {} \t Y_C{}: {} \n".format(rectangles, x_center, rectangles, y_center))
                                center = cv2.rectangle(img, (x_center, y_center), (x_center+1, y_center+1), (255, 255, 0), 5)
                                # print("SUM_X: {}".format(sum_x))
                                # print("SUM_Y: {}".format(sum_y))
                                # # if(len(centers) > 2):
                                #     print("1: {} 2: {}".format(centers[0][0], centers[1][0]))
                                #     line = cv2.line(img, (centers[0][0], centers[0][1]), (centers[1][0], centers[1][1]), (0, 0, 255), 2)
                                # for i in centers:
                                #     print(i[0])
                                #print("1: {} 2: {}".format(centers[0], centers[1]))
                                # cv2.line(img, (x_center, y_center), (, ), (0, 0, 255), 2)

                cv2.imshow("Frame", img)
                img = cv2.flip(img,1)
                cv2.imshow("Blue",res_blue)
                # cv2.imshow("Green",res_green)


                if cv2.waitKey(10) & 0xFF == 32:
                        cap.release()
                        cv2.destroyAllWindows()
                        break
if __name__=="__main__":
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_COMPLEX
        stop = 2
        main()
