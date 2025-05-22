import camera
import hand_detection
import queue
import threading

frame_queue = queue.Queue(maxsize=10)
landmark_queue = queue.Queue(maxsize=10)
stop_condition = threading.Event()

#video = camera.Capture()
#video_thread = threading.Thread(target=video.start_video, args=(frame_queue, landmark_queue, stop_condition))
#video_thread.start()

hand_detect = hand_detection.HandDetection()
hand_thread = threading.Thread(target=hand_detect.start_detection, args=(frame_queue, landmark_queue, stop_condition))
hand_thread.start()
#hand_detect.start_detection(frame_queue, landmark_queue, stop_condition)

#video_thread.join()
#hand_thread.join()

#video.end_video()