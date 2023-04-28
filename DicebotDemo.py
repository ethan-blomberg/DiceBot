import cv2 as cv
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

print('Successfully imported modules')

def take_pic():
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
	
	video_capture = cv.VideoCapture(gstreamer_pipeline(), cv.CAP_GSTREAMER)

	start = time.time()
	while(time.time() - start < 1):
		ret_val, frame = video_capture.read()

	ret_val, frame = video_capture.read()
	video_capture.release()
	return frame

def preprocess(img):	
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)	
    blur = cv.GaussianBlur(gray,(7,7),0)

    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv.SimpleBlobDetector_Params() 
    params.filterByArea = True
    params.minArea = 11000 #1310
    params.maxArea = 25500 #22700 
    detector = cv.SimpleBlobDetector_create(params)
            
    # Detect blob
    blur_not = cv.bitwise_not(blur)
    keypoints = detector.detect(blur_not)
    
    while (keypoints == []) :
        print('No keypoints found')
        cv.imshow('Blobs', blobs)
        cv.waitKey(0)
        keypoints = detector.detect(blur_not)	
	
    # Draw blob on our image as green circles
    blank = np.zeros((1, 1)) 
    blobs = cv.drawKeypoints(img, keypoints, blank, (0, 255, 0),
                    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Get blob coords
    x = int(keypoints[0].pt[0]) 
    y = int(keypoints[0].pt[1])

    # Crop down to 128p
    r = 64 #82		
    xlb = x-r
    xub = x+r
    ylb = y-r
    yub = y+r
    crop_bounds = [xlb,xub,ylb,yub]
    cropped = img[ylb:yub, xlb:xub]

    gray = cv.cvtColor(cropped,cv.COLOR_BGR2GRAY)	
    sm_img = cv.cvtColor(gray,cv.COLOR_GRAY2RGB)

    cv.imshow('img',sm_img)

    sm_img = sm_img/255.0
    sm_img = transforms.functional.to_tensor(sm_img).to(device).half()

    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
    sm_img.sub_(mean[:, None, None]).div_(std[:, None, None])
    sm_img = sm_img[None, ...]

    return sm_img

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 20)
model.load_state_dict(torch.load('renet18_0.pth'))
device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()

print('Model setup complete')

for trials in range(50):
	img = take_pic()
	print('Preprocessing')
	img = preprocess(img)
	print('Inferencing')
	y = model(img)
	y = F.softmax(y, dim=1)
	y = y.flatten()
	prediction = int(y.argmax())
	confidence = float(y[prediction])
	label = [1,10,11,12,13,14,15,16,17,18,19,2,20,3,4,5,6,7,8,9]
	print('Prediction: ' + str(label[prediction]) + ' Confidence: ' + str(confidence))
	cv.waitKey(0)
	cv.destroyAllWindows()
