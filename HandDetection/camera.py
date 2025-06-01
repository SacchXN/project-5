import queue
import threading
import cv2 as cv

# Default values
CAMERA_FRAME_WIDTH = cv.CAP_PROP_FRAME_WIDTH
CAMERA_FRAME_HEIGHT = cv.CAP_PROP_FRAME_HEIGHT

class Capture:
    def __init__(self, width = CAMERA_FRAME_WIDTH, height = CAMERA_FRAME_HEIGHT):
        self.__camera = cv.VideoCapture(0)
        self.__width = width
        self.__height = height

    def start_video(self, camera_queue: queue.Queue, support_queue: queue.Queue, stop_condition: threading.Event):
        if not self.__camera.isOpened():
            print("Error starting camera.")
            exit()

        while not stop_condition.is_set():
            ret, frame = self.__camera.read()

            if not ret:
                print("Error receiving frames.")
                continue

            # NOTE ON TIMEOUT
            # The timeout value actually stops the execution of the whole while loop therefore creating a video
            # "freeze" of the duration of the timeout. It needs to be set as low as possible otherwise it doesn't
            # feel smooth enough.

            try:
                support_queue.put(frame, timeout=0.1)
            except queue.Full:
                print('Support queue is full, skipped frame.')
                pass

            try:
                camera_queue.put(frame, timeout=0.1)
            except queue.Full:
                print('Camera queue is full, skipped frame.')

    def end_video(self):
        self.__camera.release()
        cv.destroyAllWindows()