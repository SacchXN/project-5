import os
import re
import pickle
import cv2 as cv
from HandDetection import hand_detection

# TODO: improve the gestures-label-gestures dictionaries?

# Gestures-label dictionary
gestures = {
    'open': 0,
    'claw': 1,
    'v': 2
}

# Labels-gestures dictionary
labels = {
    0: 'open',
    1: 'claw',
    2: 'v'
}

if __name__ == "__main__":
    # Define path for videos to extract frames from for the dataset
    path = r'..\..\project-5_videos'

    detection = hand_detection.HandDetection()
    videos = os.listdir(path)
    landmarks_collection = []

    for path in [os.path.join(path, file_name) for file_name in videos]:

        idx = -1
        for key in gestures:
            if re.match(f'{key}[0-9]\\.mp4', path.split('\\')[-1]):
                idx = gestures[key]

        if idx == -1:
            print("Loaded video doesn't have any expected gesture.\n")
            continue

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
        with open(r'..\..\project-5_dataset\landmarks_collection.pkl', 'wb') as f:
            pickle.dump(landmarks_collection, f)
    except Exception as e:
        print(f'Error saving pickle data: {e}')