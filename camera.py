import queue
import cv2 as cv
import threading

# Default values
CAMERA_FRAME_WIDTH = cv.CAP_PROP_FRAME_WIDTH
CAMERA_FRAME_HEIGHT = cv.CAP_PROP_FRAME_HEIGHT

class Capture:
    def __init__(self, width = CAMERA_FRAME_WIDTH, height = CAMERA_FRAME_HEIGHT):
        self.__camera = cv.VideoCapture(0)
        self.__width = width
        self.__height = height

    def start_video(self, frame_queue: queue.Queue, landmark_queue: queue.Queue, stop_condition: threading.Event):
        if not self.__camera.isOpened():
            print("Error starting camera.")
            exit()

        while not stop_condition.is_set():
            ret, frame = self.__camera.read()

            if not ret:
                #("Error receiving frames.")
                continue

            # The timeout value actually stops the execution of the whole while loop therefore creating a video
            # "freeze" of the duration of the timeout. It needs to be set as low as possible otherwise it doesn't
            # feel smooth enough

            try:
                frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                #print('Frame queue is full, skipped frame.')
                pass

            #    In case a greyscale image is required
            #    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow('Frame', frame)

            #if landmark_queue.empty():
            #    print('Landmark queue is empty.')
            #    continue

            #cv.imshow('Landmark', landmark_queue.get())
            if cv.waitKey(1) == ord('q'):
                stop_condition.set()
                break

    def end_video(self):
        self.__camera.release()
        cv.destroyAllWindows()