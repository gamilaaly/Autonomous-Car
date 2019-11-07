import cv2
import numpy as np
import random as rng
src = cv2.imread('mouse.jpeg')
print(src.shape)
dim = (400,400)
src = cv2.resize(src,dim, interpolation = cv2.INTER_AREA)
#######thresh from 100 to 105 removes floor and lane
###LINK https://docs.opencv.org/master/da/d0c/tutorial_bounding_rects_circles.html

########## to do filling within lane and around object
# we need to add the object contour to the green masked image
# in live stream then fill till u find an edge 


def thresh_callback(val):
    threshold = val

    ## [Canny]
    # Detect edges using Canny
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    cv2.imshow("canny", canny_output)
    ## [Canny]

    ## [findContours]
    # Find contours
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ## [findContours]

    ## [allthework]
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i]) #x,y,w,h = cv2.boundingRect(c)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        print(boundRect[i])
        
    ## [allthework]

    ## [zeroMat]
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    ## [zeroMat]

    ## [forContour]python
    # Draw polygonal contour + bonding rects + circles
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        ######for all contours on orig image
        ###color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # cv2.drawContours(src, contours_poly, i, color)
        # cv2.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])), \
        #   (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        # cv2.circle(src, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        #cv2.imshow('contours',src)
    ## [forContour]

    ## [showDrawings]
    # Show in a window
    cv2.imshow('Contours', drawing)
    f = cv2.add(drawing, src) #overlay contour on src
    cv2.imshow('f', f)


    ## [showDrawings]
#mouse thresh = 105

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_gray = cv2.blur(src_gray, (3,3))
## [setup]

## [createWindow]
# Create Window
source_window = 'Source'
cv2.namedWindow(source_window)
cv2.imshow(source_window, src)
## [createWindow]
## [trackbar]
max_thresh = 255
thresh = 100 # initial threshold
cv2.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
## [trackbar]

cv2.waitKey()