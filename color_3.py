import cv2
import numpy as np
import time
import math
import serial
import sys
import VL53L0X
import RPi.GPIO as GPIO
from time import sleep


tof = VL53L0X.VL53L0X()
tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)
timing = tof.get_timing()
if (timing < 20000):
    timing = 20000
print("Timing %d ms" % (timing/1000))
ser = serial.Serial(
    
    port='/dev/ttyAMA0',
    baudrate = 9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

#Servo setup
servoZPin = 18
servoXPin = 23
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(servoXPin, GPIO.OUT)
GPIO.setup(servoZPin, GPIO.OUT)
servoXPWM = GPIO.PWM(servoXPin, 50)
servoZPWM = GPIO.PWM(servoZPin, 50)
servoXPWM.start(0)
servoZPWM.start(0)

def setCamXAngle(angle):    #Servo Albastru 60 90 120 
    GPIO.output(servoXPin, True)
    duty = angle / 18 + 2.5
    servoXPWM.ChangeDutyCycle(duty)  
    sleep(1)
    GPIO.output(servoXPin, False)
    servoXPWM.ChangeDutyCycle(0)

def setCamZAngle(angle):    #Servo Metal 80 100 120
    GPIO.output(servoZPin, True)
    duty = angle / 18 + 2.5
    servoZPWM.ChangeDutyCycle(duty)   
    sleep(1)
    GPIO.output(servoZPin, False)
    servoZPWM.ChangeDutyCycle(0)

#Distanta dintre puncte
def Distanta_puncte(centru_x1,centru_x2,centru_y1,centru_y2): #pitag
    ipot = math.sqrt(((centru_x1-centru_x2)**2)+((centru_y1-centru_y2)**2))
    return ipot

#Get image from camera with OpenCV
def findContours(debug):
    fps, img = cap.read()
    #converting frame(img) from BGR (Blue-Green-Red) to HSV (hue-saturation-value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 55,50
    #defining the range of blue col
    low_blue = np.array([150,85,95])#[276,50.3,61.6])#[161, 155, 30]) #94, 80, 2 #94, 50, 50 #[94, 55, 50]
    high_blue = np.array([175,255,255])#[307, 35.8, 89.8])#[179, 255, 255]) #126, 255, 255
    # 138,157 ,,,,,,,,,,,150,175
    low_green = np.array([50, 50, 50])
    high_green = np.array([102, 255, 255])

    #finding the range blue colour in the image
    blue_mask = cv2.inRange(hsv, low_blue, high_blue)
    green_mask = cv2.inRange(hsv, low_green, high_green)
    # mask = cv2.inRange(hsv, low, upper)
    #Morphological transformation, Dilation
    #kernal = np.ones((5 ,5), "uint8")
    basic_mask = blue_mask + green_mask
    # blue=cv2.dilate(yellow, kernal)

    res_blue=cv2.bitwise_and(img, img, mask = blue_mask)
    res_green=cv2.bitwise_and(img, img, mask = green_mask)
    # res_mask = cv2.bitwise_and(img,img,mask = basic_mask)
    # Tracking Colour (Blue)
    (contours,hierarchy)=cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if(debug):
        cv2.imshow("Frame", img)
        img = cv2.flip(img,1)
        cv2.imshow("Blue",res_blue)
        # cv2.imshow("Green",res_green)

    return (contours,img)

#Get centers of object from image (full recognition, calls findContours)
def getCenters():
    (contours,img)=findContours(True)
    rectangle = []
    centers = []
    widths = []
    for frame, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 3000): #300
            x, y, w, h = cv2.boundingRect(contour)
            widths.append(w)
            border = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            rectangle.append(border)
            for rectangles in range(0, len(rectangle)):
                x_center = int((2*x+w)/2)
                y_center = int((2*y+h)/2)
                center_tuple = (x_center, y_center)
            centers.append(center_tuple)
    return {"centers" : centers,
            "width" : widths,
            }

#vert  30-170 cu 100 
def RotateCameraDefault():
    setCamXAngle(90)
    setCamZAngle(100)

#Roteste camera pana cand mijlocul unui marker este in centru
def RotateCameraObject(obj_x, obj_y):
    angleX = (obj_x/640)*62.2 #degrees
    setCamXAngle(angleX)
    sleep(0.5)

#Misca drona in lateral pana cand  mijlocul distantei dintre doua markere este in centru

def sortedCenters():
    dicti = getCenters()
    for i in range(0, len(dicti["centers"]) - 1):
            for j in range(i+1, len(dicti["centers"])):
                if dicti["centers"][i][0] > dicti["centers"][j][0]:
                    dicti["centers"][i], dicti["centers"][j] = dicti["centers"][j], dicti["centers"][i]
                    dicti["width"][i], dicti["width"][j] = dicti["width"][j], dicti["width"][i]
    return dicti

def GoToMiddle(x1,x2,y1,y2,w1,w2):
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        w1, w2 = w2, w1
    ok = False
    while ok is False:
        if x1 < 320 and x2 > 320:
            while not ((x1+x2)/2 - 10 >= 320 and (x1+x2)/2 + 10 <= 320):
                if (x1+w1+x2-w2)/2 < 320:
                    ser.write("R")
                else:
                    ser.write("L")
                dicti = sortedCenters()
                x1 = dicti["centers"].last()[0]
                for i in range(0, len(dicti["centers"])):
                    if dicti["centers"][i+1][0] > 320:
                        x1 = dicti["centers"][i][0]
                        x2 = dicti["centers"][i+1][0]
                        y1 = dicti["centers"][i][1]
                        y2 = dicti["centers"][i+1][1]
            ok = True
        if x1 < 320 and x2 < 320:
            while not x1 < 320 and x2 > 320:
                ser.write("L")
                dicti = sortedCenters()
                x1 = dicti["centers"][0][0]
                x2 = dicti["centers"][1][0]
                y1 = dicti["centers"][0][1]
                y2 = dicti["centers"][0][1]
        if x1 > 320 and x2 > 320:
             while not x1 < 320 and x2 > 320:
                ser.write("R")
                dicti = sortedCenters()
                size = len(dicti["centers"]) - 1
                x1 = dicti["centers"][size - 1][0]
                x2 = dicti["centers"][size][0]
                y1 = dicti["centers"][size - 1][1]
                y2 = dicti["centers"][size][1]
     


#Cauta path intre obiecte        
def Find_Middle_To_Go(centers,k,widths):
    boase = Distanta_puncte(centers[k-1][0],centers[k][0],centers[k-1][1],centers[k][1])
    print(boase)
    if int(boase) < 200:
        try:
            Find_Middle_To_Go(centers, k+1)
        except:
            print("nu exista nici un")
    else:
        print("a gasit pathul")
        GoToMiddle(centers[k-1][0],centers[k][0],centers[k-1][1],centers[k][1],widths[k-1],widths[k])


def getDistance():
    distance = tof.get_distance()
    if(distance <= 0):
        distance = -1
    if(distance > 1500):
        distance = 1500
    time.sleep(timing/1000000.00)
    return distance

def main():
    has_waypoint=0
    wait=0
    setCamXAngle(60)
    while(1):
        time.sleep(100)
    while(1):
        (contours,img)=findContours(True)
        rectangle = []
        centers = []
        widths = []
        sum_x = 0
        sum_y = 0
        for frame, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
            # x = approx.ravel()[0]
            # y = approx.ravel()[1]
            x_center = 0
            y_center = 0

            if(area > 4000): #300
                x, y, w, h = cv2.boundingRect(contour)
                widths.append(w)
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

        
        if (len(centers)==1):
            if(Distanta_puncte(centers[0][0],320,centers[0][1],240)>200):
                while(getDistance()>400):
                    ser.write("F") #.format(getDistance())
                has_waypoint=1
            elif (centers[0][0]<320 and not has_waypoint):
                RotateCameraObject(centers[0][0],centers[0][1])
                dist=getDistance()
                while(abs(dist-getDistance)<200):
                    ser.write("R") #.format(getDistance())
                has_waypoint=1 
            elif (centers[0][0]>320 and not has_waypoint):
                RotateCameraObject(centers[0][0],centers[0][1])
                dist=getDistance()
                while(abs(dist-getDistance)<200):
                    ser.write("L") #.format(getDistance())
                has_waypoint=1
        elif(len(centers)>=2 and not has_waypoint):
                Find_Middle_To_Go(centers,1,widths)
                RotateCameraDefault()
                while(getDistance>400):
                    ser.write("F")
                has_waypoint=1
        if(has_waypoint==0):
            while(getDistance>500):
                ser.write("F")

        if cv2.waitKey(10) & 0xFF == 32:
            cap.release()
            cv2.destroyAllWindows()
            tof.stop_ranging()
            servoZPWM.stop()
            servoXPWM.stop()
            GPIO.cleanup()
            break
        
        RotateCameraDefault()
        has_waypoint=0
        widths.clear()
        servoZPWM.stop()
        servoXPWM.stop()
        GPIO.cleanup()

if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX
    stop = 2
    main()
