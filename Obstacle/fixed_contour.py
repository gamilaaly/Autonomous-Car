import cv2
import numpy as np

src = cv2.imread("mouse.jpeg")
print(src.shape)
dim = (400, 400)
src = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)
# cv2.imshow('src',src)
###Lane Detection
inputImageHSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv", inputImageHSV)
lower_green = np.array([80,40,40 ])
upper_green = np.array([100, 255,255])
mask = cv2.inRange(inputImageHSV, lower_green, upper_green)
edges = cv2.Canny(mask, 200, 400)
# cv2.imshow("green mask", edges)

###Contour 
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_gray = cv2.blur(src_gray, (3, 3))
## [Canny]
# Detect edges using Canny
threshold = 105
canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
## [Canny]
## [findContours]
# Find contours
contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
## [findContours]
# Approximate contours to polygons + get bounding rects and circles
contours_poly = [None] * len(contours)
boundRect = [None] * len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(
        contours_poly[i]
    )  # x,y,w,h = cv2.boundingRect(c)
## [zeroMat]
drawing = np.zeros(
    (canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8
)
## [zeroMat]

## [forContour]python
# Draw polygonal contour + bonding rects + circles
hull = []
for i in range(len(contours)):
    color = (255,255,255)
    cv2.drawContours(drawing, contours_poly, i, color)
    cv2.rectangle(
        drawing,
        (int(boundRect[i][0]), int(boundRect[i][1])),
        (
            int(boundRect[i][0] + boundRect[i][2]),
            int(boundRect[i][1] + boundRect[i][3]),
        ),
        color,
        2,
    )
    # hull.append(cv2.convexHull(contours[i], False))
    # cv2.drawContours(drawing, hull, i, color, 1, 8)


# cv2.imshow("Contours", drawing)
drawing_gray= cv2.cvtColor(drawing,cv2.COLOR_BGR2GRAY)
print(edges.shape,"hiiiiiii", drawing_gray.shape)
combined = cv2.add(drawing_gray, edges)  
combined_pic = cv2.add(drawing, src) # overlay contour on src
# cv2.imshow("combined", combined_pic)

###Filling

####link https://stackoverflow.com/questions/45135950/how-to-fill-an-image-from-bottom-side-until-an-edge-is-detected-using-opencv

h, w = combined.shape[:2]
filled_from_bottom = np.zeros((h, w), dtype=np.uint8)
for col in range(w):
    for row in reversed(range(h)):
        if combined[row][col] < 255: filled_from_bottom[row][col] = 255
        else: break
# cv2.imshow("filled", filled_from_bottom)

###link https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
# # Copy the thresholded image.
# im_floodfill = im_th.copy()
 
# # Mask used to flood filling.
# # Notice the size needs to be 2 pixels than the image.
# h, w = im_th.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)
 
# # Floodfill from point (0, 0)
# cv2.floodFill(im_floodfill, mask, (0,0), 255)
 

###Erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(filled_from_bottom,kernel,iterations =1)
###Smoothing
blur = cv2.bilateralFilter(filled_from_bottom,9,75,75)
cv2.imshow("smoothed", blur)
###Path finding  link http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
#https://stackoverflow.com/questions/21039535/opencv-extract-path-centerline-from-arbitrary-area
imgblur = blur.copy()
skel = np.zeros(imgblur.shape,np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
size = np.size(imgblur)
while( not done):
    eroded = cv2.erode(imgblur,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(imgblur,temp)
    skel = cv2.bitwise_or(skel,temp)
    imgblur = eroded.copy()
 
    zeros = size - cv2.countNonZero(imgblur)
    if zeros==size:
        done = True

cv2.imshow("path",skel)
not_skel = cv2.bitwise_not(skel)
skel_rgb = cv2.cvtColor(skel,cv2.COLOR_GRAY2RGB)
cv2.imshow("combinedPath", cv2.add(combined_pic, skel_rgb))

cv2.waitKey()