import cv2 as cv
import numpy as np
font = cv.FONT_HERSHEY_SIMPLEX

def open_camera(capture):

    while True:
        ret,frame = capture.read()
        frame = cv.resize(frame,None,fx=1.10,fy=1.10,interpolation = cv.INTER_AREA)
       
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        dst = cv.cornerHarris(gray,2,3,0.04)
        dst = cv.dilate(dst,None)

        frame[dst>0.01*dst.max()]=[0,0,255]

        #cv.putText(frame,'ce muie de prost am detectat',(10,50), font, 1.5,(255,255,255),2,cv.LINE_AA)

        cv.imshow('frame',frame)

        c = cv.waitKey(1)
        if c == 32:
            break

    capture.release()
    cv.destroyAllWindows()

def maybe_not_open(capture):

    if not capture.isOpened():
        raise IOError("Cannot open webcam get your tape off")

if __name__=="__main__":

    capture = cv.VideoCapture(0)
    maybe_not_open(capture)
    open_camera(capture)
