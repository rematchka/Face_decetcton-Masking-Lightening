# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:51:07 2017

@author: extra
"""
"""
import cv2

import numpy as np
  
img = cv2.imread('FB_IMG_1463846848502.jpg')
cv2.imshow('image ',img)
cv2.waitKey(1)

img = cv2.medianBlur(img,5)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
 
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

                
print (circles)
      #circles = np.uint16(np.around(circles))
for i in circles[0,:]:
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
           break
cv2.destroyWindow("preview")
"""

import cv2
import numpy as np

img = cv2.imread('eye.jpg',0)
img = cv2.medianBlur(img,3)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
import cv2
import numpy as np

img = cv2.imread('FB_IMG_1463846848502.jpg')
cv2.imshow("preview", img)
cv2.waitKey(1)
img = cv2.medianBlur(img,51)
imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cimg = cv2.cvtColor(imgg,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(imgg,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
if circles is None:
    print("problem")
            
print (circles)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        
cv2.destroyWindow("preview")
"""