import cv2 as cv
import numpy as np
font = cv.FONT_HERSHEY_SIMPLEX

def nothing(x):
    pass

def trackpos(frame):
    
    cv.imshow('frame',frame)
    
    cv.createTrackbar('R','frame',0,255,nothing)
    cv.createTrackbar('G','frame',0,255,nothing)
    cv.createTrackbar('B','frame',0,255,nothing)

    switch = '0 : OFF \n1 : ON'
    cv.createTrackbar(switch, 'frame',0,1,nothing)

    r = cv.getTrackbarPos('R','frame')
    g = cv.getTrackbarPos('G','frame')
    b = cv.getTrackbarPos('B','frame')
    s = cv.getTrackbarPos(switch,'frame')
    return r,g,b,s
def open_camera(capture):

    while True:
        ret,frame = capture.read()
        frame = cv.resize(frame,None,fx=0.010,fy=0.010,interpolation = cv.INTER_AREA)
         #cv.putText(frame,'ce muie de prost am detectat',(10,50), font, 1.5,(255,255,255),2,cv.LINE_AA)

        r,g,b,s = trackpos(frame)

        c = cv.waitKey(1)
        if c == 32:
            break

        if s == 0:
            frame[:] = 0
        else:
            frame[:] = [b,g,r]

    capture.release()
    cv.destroyAllWindows()

def maybe_not_open(capture):

    if not capture.isOpened():
        raise IOError("Cannot open webcam get your tape off")

if __name__=="__main__":

    capture = cv.VideoCapture(0)
    maybe_not_open(capture)
    open_camera(capture)
