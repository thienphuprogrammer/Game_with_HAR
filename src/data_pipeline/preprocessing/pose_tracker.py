import time

import cv2
import mediapipe as mp
import numpy as np


def extract_and_zip_landmarks(landmark_list, num_landmarks, prefix) -> np.array:
    landmarks = np.empty((0, 3))
    if landmark_list:
        for i, landmark in enumerate(landmark_list.landmark):
            new_landmark = np.array([landmark.x, landmark.y, landmark.z])
            landmarks = np.vstack([landmarks, new_landmark])
    else:
        for i in range(num_landmarks):
            landmarks = np.vstack([landmarks, np.array([np.nan, np.nan, np.nan])])
    return landmarks


def extract_and_zip_landmarks_dict(landmark_list, num_landmarks, prefix) -> dict:
    landmarks = {}
    if landmark_list:
        for i, landmark in enumerate(landmark_list.landmark):
            new_landmark = (landmark.x, landmark.y, landmark.z)
            landmarks[f"{prefix}_{i}"] = new_landmark
    else:
        for i in range(num_landmarks):
            landmarks[f"{prefix}_{i}"] = (np.nan, np.nan, np.nan)
    return landmarks


class PoseTracker:
    def __init__(self):
        # Initialize the MediaPipe Face Detection and Face Mesh modules
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )
        # FPS calculation
        self.prev_time = 0
        self.current_time = 0

    def process_frame(self, frame, num_landmarks_pose=33) -> np.array:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame
        results = self.holistic.process(rgb_frame)
        row_data = extract_and_zip_landmarks(
            results.pose_landmarks, num_landmarks_pose, "pose"
        )

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
            )
        return row_data

    def process_frame_dict(self, frame, num_landmarks_pose=33) -> dict:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame
        results = self.holistic.process(rgb_frame)
        row_data = extract_and_zip_landmarks_dict(
            results.pose_landmarks, num_landmarks_pose, "pose"
        )

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
            )
        return row_data

    def get_fps(self):
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.prev_time)
        self.prev_time = self.current_time
        return fps

    def __del__(self):
        self.holistic.close()
