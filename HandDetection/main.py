import queue
import threading
import cv2 as cv
import torch
import camera
import hand_detection
from GestureRecognitionNN import neural_network
from GestureRecognitionNN.neural_network import DEVICE
from GestureRecognitionNN.dataset_building import labels

# NOTE ON CV2 AND THREADING INTERACTION:
# The stream of cv2.imshow() needs to be in the main because there can be freezes or issues if being used within
# threaded functions: cv2.imshow() and other functions (like cv2.waitKey()) called from threads can lead to race
# conditions/deadlocks/other behavior

## Possible improvements: put lock on queues to avoid race conditions?

# Support variables
camera_queue = queue.Queue(maxsize=10)
landmark_queue = queue.Queue(maxsize=10)
support_queue = queue.Queue(maxsize=10)
stop_condition = threading.Event()
detection = hand_detection.HandDetection()

# Camera thread
video = camera.Capture()
video_thread = threading.Thread(target=video.start_video,
                                args=(camera_queue, support_queue, stop_condition))
video_thread.start()

# Hand detection thread
hand_detect = hand_detection.HandDetection()
hand_thread = threading.Thread(target=hand_detect.start_detection_draw,
                               args=(landmark_queue, support_queue, stop_condition, 2))
hand_thread.start()

model = neural_network.NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load('../GestureRecognitionNN/model_weights.pth', weights_only=True))
model.eval()

# showOnScreen:
# Show camera and landmark frames on screen
while True:

    if not camera_queue.empty():
        # TODO: both streams look really laggy, probably because there are too many calculations being done in the while
        #       Possible solutions: separate thread for the gesture recognition part? Line 50 to 59
        frame = camera_queue.get()
        cv.imshow('Frame', frame)
        landmarks = detection.start_detection_landmark(frame, 1)
        if landmarks:
            temp = []
            for landmark in landmarks[0]:  # landmarks[0] being the landmarks of the first and only hand
                temp.append(landmark.x)
                temp.append(landmark.y)
                temp.append(landmark.z)
            X = torch.tensor(temp).unsqueeze(0).float()
            pred = model(X)
            print(f'Prediciton: {labels[pred.argmax(1).item()]}')

    if not landmark_queue.empty():
        cv.imshow('Landmark', landmark_queue.get())

    if cv.waitKey(1) == ord('q'):
        stop_condition.set()
        break

# Join threads
video_thread.join()
hand_thread.join()

# Close camera stream
video.end_video()
