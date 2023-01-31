import cv2 as cv #version 4.1.1
import numpy as np
import time
import imutils
import matplotlib.pyplot as plt

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
window_title = "Camera V2"

video_capture = cv.VideoCapture(gstreamer_pipeline(), cv.CAP_GSTREAMER)

start = time.time()
while(time.time() - start < 1):
	ret_val, frame = video_capture.read()

ret_val, frame = video_capture.read()

gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)	
blur = cv.GaussianBlur(gray,(3,3),0)
edge = cv.Canny(blur,50,200,255)

detected_circles = cv.HoughCircles(
	edge,
	cv.HOUGH_GRADIENT,
	1,
	20,
	param1 = 50,
	param2 = 30,
	minRadius = 80,
	maxRadius = 120
)

if detected_circles is not None:
	detected_circles = np.uint16(np.around(detected_circles))
	for pt in detected_circles[0, :]:
		a, b, r = pt[0], pt[1], pt[2]
		cv.circle(frame, (a,b), r, (0,255,0),2)
		break

xlb = a-r
xub = a+r
ylb = b-r
yub = b+r
cropped = frame[ylb:yub, xlb:xub]
cropped_blur = blur[ylb:yub, xlb:xub]
cv.imshow("%2d by %2d" % ((2*r),(2*r)), cropped)
cv.waitKey(0)

lb = r-50
ub = r+50
cropped = cropped[lb:ub , lb:ub]
cropped_blur = cropped_blur[lb:ub , lb:ub]
cropped_edge = cv.Canny(cropped_blur,50,200,0)
cropped_edge= cv.GaussianBlur(cropped_edge,(3,3),0)

detected_center = cv.HoughCircles(
	cropped_edge,
	cv.HOUGH_GRADIENT,
	1,
	20,
	param1 = 50,
	param2 = 30,
	minRadius = 15,
	maxRadius = 24
)

if detected_center is not None:
	detected_center = np.uint16(np.around(detected_center))
	for pt in detected_center[0, :]:
		a2, b2, r2 = pt[0], pt[1], pt[2]
		cv.circle(cropped, (a2,b2), r2, (0,255,0),2)
		cv.circle(cropped, (a2,b2), 1, (0,255,0),3)
		print("Radius: %3d, Pt: %3d, %3d" % (r2, a2, b2))
		
		xlb = a2-r2
		xub = a2+r2
		ylb = b2-r2
		yub = b2+r2
		
		cropped = cropped[ylb:yub, xlb:xub]
		cv.imshow("detected center", cropped)
		cv.waitKey(0)
		
		xlb = a-50+a2-r2
		xub = a-50+a2+r2
		ylb = b-50+b2-r2
		yub = b-50+b2+r2
		main_crop = frame
		cv.imshow("main_crop", main_crop)
		cv.waitKey(0)
		
		break
else:
	print('No center found')
	cv.imshow("No center found", cropped_edge)
	cv.waitKey(0)
