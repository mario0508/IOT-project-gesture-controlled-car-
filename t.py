import cv2
import numpy as np
import copy
import math
#import socket
#print("enter port:-")
#port=input()
#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.bind(('192.168.43.211', port))
#s.listen(5)
#clientsocket, address = s.accept()

cap_region_x_begin=0.5 
cap_region_y_end=0.8  
threshold = 60  
blurValue = 41  
bgSubThreshold = 50
learningRate = 0


isBgCaptured = 0   
triggerSwitch = False  

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None): 
            cnt = 0
            for i in range(defects.shape[0]):  
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  
                if angle <= math.pi / 2: 
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  
    frame = cv2.flip(frame, 1)  
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    
    if isBgCaptured == 1:  
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] 
        cv2.imshow('mask', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)


       
        thresh1 = copy.deepcopy(thresh)
        _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)+1
        maxArea = -1
        if length > 0:
            for i in range(length):  
             temp = contours[i]
             area = cv2.contourArea(temp)
             if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt = calculateFingers(res,drawing)
            print(isFinishCal,cnt+1)
	    #clientsocket.send(bytes(cnt+1))
    		
            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print (cnt)
			

       	cv2.imshow('output', drawing)
    k = cv2.waitKey(10)
    if k == 27:  
        camera.release()
        cv2.destroyAllWindows()
        break
    ## Comment tilak:- Background substractor code
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'): 
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
print ('!!!Trigger On!!!')
