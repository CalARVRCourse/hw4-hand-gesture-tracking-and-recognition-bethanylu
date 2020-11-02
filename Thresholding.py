# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:31:23 2020

@author: bethany
"""

from __future__ import print_function
import cv2
import argparse
import numpy as np
import pyautogui
import keyboard
import math

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'
window_name = 'Threshold Demo'
isColor = False

#cam = cv2.VideoCapture(0)
#def nothing(x):  
 #   pass  
#cv2.namedWindow(window_name)    
# Create a Trackbar to choose a value for a parameter    
#cv2.createTrackbar(parameter_value_name, window_name , parameter_min_value, parameter_max_value, nothing)


def Threshold_Demo(val):
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
    cv2.imshow(window_name, dst)
    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        cv2.destroyAllWindows()
        cam.release()
        return(True)
    return(False)

def nothing(x):
    pass
    

cam = cv2.VideoCapture(0)
cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_type, window_name , 3, max_type, nothing)
# Create Trackbar to choose Threshold value
cv2.createTrackbar(trackbar_value, window_name , 0, max_value, nothing)
# Call the function to initialize
cv2.createTrackbar(trackbar_blur, window_name , 1, 20, nothing)
# create switch for ON/OFF functionality
color_switch = 'Color'
cv2.createTrackbar(color_switch, window_name,0,1,nothing)
cv2.createTrackbar('Contours', window_name,0,1,nothing)

  
  

count = 0
previous = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    blur_value = cv2.getTrackbarPos(trackbar_blur, window_name)
    blur_value = blur_value+ (  blur_value%2==0)
    isColor = (cv2.getTrackbarPos(color_switch, window_name) == 1)
    findContours = (cv2.getTrackbarPos('Contours', window_name) == 1)
    
    #convert to grayscale
    if isColor == False:
        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
        blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
        if findContours:
            _, contours, hierarchy = cv2.findContours( blur, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
            blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  #add this line
            output = cv2.drawContours(blur, contours, -1, (0, 255, 0), 1)
            print(str(len(contours))+"\n")
        else:
            output = blur
        
        
    else:
        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
        blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
        output = blur
        
    lower_HSV = np.array([0, 40, 0], dtype = "uint8")  
    upper_HSV = np.array([7, 200, 255], dtype = "uint8")  
      
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  
      
      
    lower_YCrCb = np.array((0, 40, 0), dtype = "uint8")  
    upper_YCrCb = np.array((55, 255, 255), dtype = "uint8")  
          
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  
      
    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)  

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 1)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 1)  
      
    # blur the mask to help remove noise, then apply the  
    # mask to the frame  
    skinMask = cv2.GaussianBlur(skinMask, (1, 1), 0) 
    skin = cv2.bitwise_and(frame, frame, mask = skinMask) 
    
    gray = cv2.cvtColor(skin,cv2.COLOR_BGR2GRAY)  
    
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )  

    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(gray,ltype=cv2.CV_16U)  
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    blank_ch = 255*np.ones_like(label_hue)  
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0  
    conImg = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

    statsSortedByArea = stats[np.argsort(stats[:, 4])]  
    
    if (ret>2):  
        try:  
            roi = statsSortedByArea[-3][0:4]  
            x, y, w, h = roi  
            subImg = labeled_img[y:y+h, x:x+w]  
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            maxCntLength = 0  
            for i in range(0,len(contours)):  
                cntLength = len(contours[i])  
                if(cntLength>maxCntLength):  
                    cnt = contours[i]  
                    maxCntLength = cntLength  
            if(maxCntLength>=5):  
                ellipseParam = cv2.fitEllipse(cnt)  
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
              
            #handContour = max(contours, key = lambda x: cv2.contourArea(x))
            
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)  
            #cv2.imshow("ROI "+str(2), subImg)  
            cv2.waitKey(1)  
        except:  
            print("No hand found")  
            
    else:
        roi = statsSortedByArea[-3][0:4]  
        x, y, w, h = roi  
        subImg = labeled_img[y:y+h, x:x+w] 

        _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
        ellipseParam = cv2.fitEllipse(contours[0])  
        subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
        subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)
          
    #(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)

    _, contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
    contours=sorted(contours,key=cv2.contourArea,reverse=True)       
    if len(contours)>1:  
        #print('contour>1')
        largestContour = contours[0]  
        hull = cv2.convexHull(largestContour)
        handArea = cv2.contourArea(largestContour)
        hullArea = cv2.contourArea(hull)
        hull = cv2.convexHull(largestContour, returnPoints = False)
        ratio = (hullArea - handArea/handArea)/100
        fingerCount = 0 
        for cnt in contours[:1]:  
            defects = cv2.convexityDefects(cnt,hull)  
            if(not isinstance(defects,type(None))):  
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])  
                    
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)  
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)  
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)  
                    s = (a+b+c)/2
                    ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                    
                    d = (2*ar)/a
                    
                    angle = np.arccos((b**2 + c**2  - a**2 ) / (2*b*c)) * 57    
                    
                    if angle <= 90 and d > 30:
                        fingerCount += 1
                        conImg = cv2.circle(frame, far, 4, [0, 0, 255], -1)  

 
                    conImg = cv2.line(frame,start,end,[0,255,0],2) 
    
    fingerCount += 1                
    M = cv2.moments(largestContour)  
    offsetX = 660
    offsetY = 350
    scaleX = 1
    scaleY = 1
    cX = offsetX + scaleX *int(M["m10"] / M["m00"])  
    cY = offsetY + scaleY *int(M["m01"] / M["m00"])  
    pyautogui.moveTo(cX, cY, duration=0.001)
    
    
    current = fingerCount
    print('count', count)
    print ('previous', previous)
    print('current', current)
    if count > 0:
        if previous == 5 and current == 1:
            print ('exit')
            pyautogui.press('esc')
            escPressed = True
            pyautogui.press('esc')
            escPressed = True
    count += 1
    previous = fingerCount

                
    spacePressed = keyboard.is_pressed('space')
    if(fingerCount == 1 and not(spacePressed)):       
        pyautogui.press('space')  
        spacePressed = True      
    else:  
        spacePressed = False  
        
    handRingArea = statsSortedByArea[-3][0:4]  
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #fingerCount = 6
    if fingerCount ==1:
        if ratio > 200:
            cv2.putText(conImg, 'one', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(conImg, 'zero', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif fingerCount == 2:
        cv2.putText(conImg, 'two', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif fingerCount == 3:
        cv2.putText(conImg, 'three', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif fingerCount == 4:
        cv2.putText(conImg, 'four', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif fingerCount == 5:
        cv2.putText(conImg, 'high five!', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        

    
    
    cv2.imshow(window_name, conImg)
    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        cv2.destroyAllWindows()
        cam.release()
        break
