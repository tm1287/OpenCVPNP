import numpy as np
import cv2
import glob
import time
import json
import math


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.


cap = cv2.VideoCapture(1)

with open("data.json") as f:
            json_data = json.load(f)
            cameraMatrix = np.array(json_data["camera_matrix"])
            distortionMatrix = np.array(json_data["distortion"])
			
def compute_output_values(rvec, tvec):
	x = tvec[0][0]
	z = tvec[2][0]
	print(x,z)
	# distance in the horizontal plane between camera and target
	distance = math.sqrt(x**2 + z**2)
	# horizontal angle between camera center line and target
	angle1 = math.atan2(x, z)
	rot, _ = cv2.Rodrigues(rvec)
	rot_inv = rot.transpose()
	pzero_world = np.matmul(rot_inv, -tvec)
	angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])
	return distance, angle1*(180/math.pi), angle2*(180/math.pi)


while(True):
	
# Capture frame-by-frame
	ret, frame = cap.read()

# Our operations on the frame come here
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
# If found, add object points, image points (after refining them)

	if ret == True:
		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.
		objpoints.append(objp)

		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)

		# Draw and display the corners
		frame = cv2.drawChessboardCorners(frame, (9,6), corners2,ret)
		retval, rvec, tvec = cv2.solvePnP(objp.reshape(54,3,1),corners2.reshape(54,2,1), cameraMatrix, distortionMatrix)
		outputs = compute_output_values(rvec, tvec)
		coord_string = "{0:.2f}".format(outputs[0]) + "   " + "{0:.2f}".format(outputs[1]) + "   " + "{0:.2f}".format(outputs[2])
		#coord_string = "{3.4f} {3.4f} {3.4f}".format(outputs[0])
		cv2.putText(frame, coord_string, (0,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))


		
	# Display the resulting frame
	cv2.imshow('frame',frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cv2.destroyAllWindows()


