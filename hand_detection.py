import queue
import threading
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# class HandDetection:
#     def __init__(self):
#         self.__model_path = r'./hand_landmarker.task'
#         self.__BaseOptions = mp.tasks.BaseOptions
#         self.__HandLandmarker = mp.tasks.vision.HandLandmarker
#         self.__HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
#         self.__VisionRunningMode = mp.tasks.vision.RunningMode
#
#     def start_detection(self, frame_queue: queue.Queue, landmark_queue: queue.Queue, stop_condition: threading.Event):
#         options = self.__HandLandmarkerOptions(
#             base_options=self.__BaseOptions(model_asset_path=self.__model_path),
#             running_mode=self.__VisionRunningMode.IMAGE)
#         landmarker = self.__HandLandmarker.create_from_options(options)
#
#         #while not stop_condition.is_set():
#         #    if frame_queue.empty():
#         #        print('Frame queue is empty.')
#         #    continue
#
#         #frame = frame_queue.get()
#         #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy.asarray(frame))
#         #mp_image = mp.Image.create_from_file(r'C:\Users\sacch\Documents\GitHub\project-5\istockphoto-878155798-612x612.jpg')
#         mp_image = cv.imread(r'C:\Users\sacch\Documents\GitHub\project-5\istockphoto-878155798-612x612.jpg')
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_image)
#         result = landmarker.detect(mp_image)
#
#         annotated_image = draw_landmarks_on_image(mp_image, result)
#
#         try:
#             cv2.imshow('Landmark', annotated_image)
#         except Exception as e:
#             print(e)
#         #try:
#         #   landmark_queue.put(annotated_image, timeout=0.1)
#         #except queue.Full:
#         #   print('Landmark queue is full, skipped frame.')

class HandDetection:
    def __init__(self, path: str = r'./hand_landmarker.task'):
        self.__base_options = python.BaseOptions(model_asset_path = path)
        self.__landmarker = None

    def start_detection(self, frame_queue: queue.Queue, landmark_queue: queue.Queue, stop_condition: threading.Event,
                        num_hands: int = 2):
        options = vision.HandLandmarkerOptions(
            base_options=self.__base_options,
            num_hands=num_hands)

        self.__landmarker = vision.HandLandmarker.create_from_options(options)
        image = mp.Image.create_from_file(r"C:\Users\sacch\Desktop\ele4.jpg")
        detection_result = self.__landmarker.detect(image)
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow('video', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        pold = 1
