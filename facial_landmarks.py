# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the required packages
import numpy as np
import face_utils
import argparse
import imutils
import dlib
import cv2

# constructing the argument parser and parsing the arguments
# see the `argparse` documentation for detailed information
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required = True,
	help = "path to facial landmark predictor")
ap.add_argument("-i", "--image", required = True,
	help = "path to input image")
# --shape-predictor : This is the path to dlib’s pre-trained facial
#                     landmark detector.
# --image : The path to the input image that we want to detect facial
#			landmarks on.
args = vars(ap.parse_args())

#-----------------------------------------------------------------------------

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Line 26 initializes dlib’s pre-trained face detector based on a modification
# to the standard Histogram of Oriented Gradients + Linear SVM method for
# object detection.

# Line 27 then loads the facial landmark predictor using the path to the
# supplied --shape-predictor .


#-----------------------------------------------------------------------------

# But before we can actually detect facial landmarks, we first need to detect
# the face in our input image

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# Line 43 loads our input image from disk via OpenCV, then pre-processes the
# image by resizing to have a width of 500 pixels and converting it to
# grayscale (Lines 44 and 45).
# Line 48 handles detecting the bounding box of faces in our image.

#-----------------------------------------------------------------------------

# Given the (x, y)-coordinates of the faces in the image, we can now apply
# facial landmark detection to each of the face regions

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, the
	# convert the facial landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i+1), (x-10, y-10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks and
	# draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

#show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)

# We start looping over each of the face detections on Line 61.

# For each of the face detections, we apply facial landmark detection 
# on Line 64, giving us the 68 (x, y)-coordinates that map to the specific
# facial features in the image.

# Line 65 then converts the dlib shape  object to a NumPy array with
# shape (68, 2).

# Lines 69 and 70 draw the bounding box surrounding the detected face
# on the image  while Lines 73 and 74 draw the index of the face.

# Finally, Lines 78 and 79 loop over the detected facial landmarks and
# draw each of them individually.

# Lines 82 and 83 simply display the output image  to our screen.

# Lines 53 and 54 simply display the output image  to our screen.

#-----------------------------------------------------------------------------