import os
import pickle
import cv2 as cv
from HandDetection import hand_detection

# Define path for videos to extract frames from for the dataset
path = '../../videos'

detection = hand_detection.HandDetection()
landmarks_collection = []
videos = os.listdir(path)

# TODO: improve dataset building where there are more than one videos for same gesture i.e.:open_1.mp4, open_2.mp4, ...
# Possible solution: dict with keys being gesture names (open, claw, gun, ...) and values being labels (0, 1, 2, ...)
#                    and check file names with regex (str.match('claw.*'))

for idx, path in enumerate([os.path.join(path, file_name) for file_name in videos]):
    print(f"Index: {idx}, path: {path}")
    cap = cv.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print('No frame from video.')
            break

        landmarks = detection.start_detection_landmark(frame, 1)

        if landmarks:
            temp = []
            for landmark in landmarks[0]:  # landmarks[0] being the landmarks of the first and only hand
                temp.append(landmark.x)
                temp.append(landmark.y)
                temp.append(landmark.z)
            temp.append(idx)
            landmarks_collection.append(temp)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# Saving the dataset locally
try:
    with open('landmarks_collection.pkl', 'wb') as f:
        pickle.dump(landmarks_collection, f)
except Exception as e:
    print(f'Error saving pickle data: {e}')


