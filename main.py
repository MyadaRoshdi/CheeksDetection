# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np

import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)


    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        '''
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            '''

        # Draw rectangle on each face
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

       # Draw left cheek
        cv2.rectangle(frame,
                      (shape[1][0], shape[1][1]),
                      (shape[49][0], shape[49][1]),
                      (0,0,255),
                      1)
        # Crop the left cheek
        crop_left_cheek = frame[shape[1][0]:shape[49][0], shape[1][1]:shape[49][1]]

        # Get average pixel values for left cheek
        l_cheek_pix= np.average(crop_left_cheek)


        # Draw right cheek
        cv2.rectangle(frame,
                      (shape[54][0], shape[54][1]),
                      (shape[15][0], shape[15][1]),
                      (0,0,255),
                      1)

        # Crop the right cheek
        crop_right_cheek = frame[shape[54][0]:shape[15][0], shape[54][1]:shape[15][1]]

        # Get average pixel values for right cheek
        r_cheek_pix= np.average(crop_right_cheek)

        # Display average pixel value for right cheek on video frame
        cv2.putText(frame, 'avg_pix_l = ' + str(round(l_cheek_pix, 2)) + '\n avg_pix_r = ' + str(round(l_cheek_pix, 2)),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0, 255, 0), 1)
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
