import queue
import threading
import cv2 as cv
import camera
import hand_detection


# NOTE ON CV2 AND THREADING INTERACTION:
# The stream of cv2.imshow() needs to be in the main because there can be freezes or issues if being used within
# threaded functions: cv2.imshow() and other functions (like cv2.waitKey()) called from threads can lead to race
# conditions/deadlocks/other behavior

# Support variables
camera_queue = queue.Queue(maxsize=10)
landmark_queue = queue.Queue(maxsize=10)
support_queue = queue.Queue(maxsize=10)
stop_condition = threading.Event()

# Camera thread
video = camera.Capture()
video_thread = threading.Thread(target=video.start_video, args=(camera_queue, support_queue, stop_condition))
video_thread.start()

# Hand detection thread
hand_detect = hand_detection.HandDetection()
hand_thread = threading.Thread(target=hand_detect.start_detection, args=(landmark_queue, support_queue, stop_condition,
                                                                         2))
hand_thread.start()

# Show camera and landmark frames on screen
while True:

    if not camera_queue.empty():
        cv.imshow('Frame', camera_queue.get())
        #print(1)

    if not landmark_queue.empty():
        cv.imshow('Landmark', landmark_queue.get())
        #print(2)

    if cv.waitKey(1) == ord('q'):
        stop_condition.set()
        break

# Join threads
video_thread.join()
hand_thread.join()

# Close camera stream
video.end_video()
