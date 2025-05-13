import cv2 as cv

# Default values
CAMERA_FRAME_WIDTH = cv.CAP_PROP_FRAME_WIDTH
CAMERA_FRAME_HEIGHT = cv.CAP_PROP_FRAME_HEIGHT

class Capture:
    def __init__(self, width = CAMERA_FRAME_WIDTH, height = CAMERA_FRAME_HEIGHT):
        self.__camera = cv.VideoCapture(0)
        self.__width = width
        self.__height = height

    def start_video(self):
        if not self.__camera.isOpened():
            print("Error starting camera.")
            exit()

        while True:
            ret, frame = self.__camera.read()

            if not ret:
                print("Error receiving frames.")
                break

            #    In case a greyscale image is required
            #    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow('Video', frame)

            if cv.waitKey(1) == ord('q'):
                break

    def end_video(self):
        self.__camera.release()
        cv.destroyAllWindows()