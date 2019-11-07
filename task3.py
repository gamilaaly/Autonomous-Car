import numpy as np
import cv2
import logging
import serial
import time
import requests as req
import json
import math

def crop_lane(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (int(0.2* width) , int(height*0.4)),
        (int(width*0.8),int(height*0.4)),
        (int(width*0.8),int(height)),
        (int(0.2*width), int(height)),
    ]], np.int32)

    # polygon = np.array([[
    #     (0, height),
    #     (width , int(height * 0.4)),
    #     (width , int(height)),
    #     (0, int(height)),
    # ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    # cv2.imshow("cropped", cropped_edges)
    return cropped_edges

    return


def contour(src,edges ,thresh = 105 ):
    ###Contour
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))
    # src_gray = crop_lane(src_gray)
    src_gray = region_of_interest(src_gray)
    ## [Canny]
    # Detect edges using Canny
    threshold = thresh
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
        print("coordinates",boundRect[i])
    ## [zeroMat]
    drawing = np.zeros(
        (canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8
    )
    ## [zeroMat]

    ## [forContour]python
    # Draw polygonal contour + bonding rects + circle
    for i in range(len(contours)):
        color = (255, 255, 255)
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

    cv2.imshow("Contours", drawing)
    drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    combined = cv2.add(drawing_gray, edges)
    combined_pic = cv2.add(drawing, src)  # overlay contour on src
    #cv2.imshow("combined", combined_pic)

    h, w = combined.shape[:2]
    filled_from_bottom = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        for row in reversed(range(h)):
            if drawing_gray[row][col] < 255:
                filled_from_bottom[row][col] = 255
            else:
                break

    cv2.imshow("filled", filled_from_bottom)

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
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(filled_from_bottom, kernel, iterations=1)
    # ###Smoothing
    blur = cv2.bilateralFilter(filled_from_bottom, 9, 75, 75)
    # cv2.imshow("smoothed", blur)
    # ###Path finding  link http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
    # # https://stackoverflow.com/questions/21039535/opencv-extract-path-centerline-from-arbitrary-area
    imgblur = blur.copy()
    skel = np.zeros(imgblur.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    size = np.size(imgblur)
    while (not done):
        eroded = cv2.erode(imgblur, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(imgblur, temp)
        skel = cv2.bitwise_or(skel, temp)
        imgblur = eroded.copy()

        zeros = size - cv2.countNonZero(imgblur)
        if zeros == size:
            done = True
    # not_skel = cv2.bitwise_not(skel)
    skel_rgb = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    pic = cv2.add(combined_pic, skel_rgb)
    return pic




def detect_edges(frame):
    inputImageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   # cv2.imshow("hsv", inputImageHSV)
    lower_green = np.array([80, 40, 40])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(inputImageHSV, lower_green, upper_green)
    edges = cv2.Canny(mask, 200, 400)
    return edges


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 0.2),
        (width, height * 0.2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    #cv2.imshow("cropped", cropped_edges)
    return cropped_edges


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                    np.array([]), minLineLength=8, maxLineGap=4)
    return line_segments


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


def detect_lane(frame):
    edges = detect_edges(frame)
    ROI = region_of_interest(edges)
    line_segments = detect_line_segments(ROI)
    lane_lines = average_slope_intercept(frame, line_segments)

    return lane_lines, edges


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def compute_steering_angle(frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
                    We assume that camera is calibrated to point to dead center
                """
    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        # Do Nothing
    else:
        height, width, _ = frame.shape
        if len(lane_lines) == 1:
            logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
            x1, _, x2, _ = lane_lines[0][0]
            x_offset = x2 - x1
        else:
            _, _, left_x2, _ = lane_lines[0][0]
            _, _, right_x2, _ = lane_lines[1][0]
            camera_mid_offset_percent = 0.02  # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
            mid = int(width / 2 * (1 + camera_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid
        # find the steering angle, which is angle between navigation direction to end of center line
        y_offset = int(height / 2)
        angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
        steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel
        return steering_angle

def send_motion_commands(s, steering_angle):
    if steering_angle >= 87 and steering_angle <= 93:
        s.write(b"f")
        print("forward")
        #time.sleep(0.25)
    elif steering_angle < 87:
        s.write(b"l")
        print("left")
        #time.sleep(0.25)
    elif steering_angle == None:
        s.write(b"f")
        print("stop")
        #time.sleep(0.25)
    else:
        s.write(b"r")
        print("right")

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    steering_angle_radian = steering_angle / 180 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def main():
    url = "http://172.28.130.47:8080/shot.jpg"
    resp = req.get("http://78608f40.ngrok.io/getRoom")
    count = 1
    json_string = json.loads(resp.text)
    print("I am not in yet !!")
    if json_string['t'] == 2:
        s = serial.Serial('COM13', 9600, timeout=1)
        print('hello from the other side')
        while 1:
                img_resp = req.get(url)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                inputImage = cv2.imdecode(img_arr, -1)

                if count % 5 == 0:
                    inputImage = contour(inputImage, edges, 200)
                else:
                    lane_lines, edges = detect_lane(inputImage)
                lane_lines_image = display_lines(inputImage, lane_lines)
                steering = compute_steering_angle(inputImage, lane_lines)
                if steering == None:
                    print("None")
                    s.write(b"f")
                    #time.sleep(0.25)

                else:
                    heading_line_image = display_heading_line(inputImage, steering, (0, 0, 255), 5, )
                    send_motion_commands(s, steering)
                    cv2.imshow("AndroidCam", inputImage)
                    #cv2.imshow("Lane Lines", lane_lines_image)
                    cv2.imshow("Heading Line", heading_line_image)
                if cv2.waitKey(1) == 27:
                    break

                count += 1


main()
