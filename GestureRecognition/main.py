import cv2 as cv
from HandDetection import hand_detection

# DATASET CREATION

## TO DO:
# Manage dataset created: save it locally? directly send it to the NN?

# Cap needs to be cycled for each video available to extract frames
detection = hand_detection.HandDetection()
landmarks_collection = []

for idx, path in enumerate(['../open.mp4', '../close.mp4']):
    cap = cv.VideoCapture('../open.mp4')

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print('No frame from video.')
            break

        landmarks = detection.start_detection_landmark(frame, 1)

        if landmarks:
            temp = []
            for landmark in landmarks[0]: # landmarks[0] being the landmarks of the first and only hand
                temp.append(landmark.x)
                temp.append(landmark.y)
                temp.append(landmark.z)
            temp.append(idx)
            landmarks_collection.append(temp)

        #cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()