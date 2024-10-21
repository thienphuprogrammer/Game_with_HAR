import os

import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm import tqdm

from src.data_pipeline.preprocessing.pose_tracker import PoseTracker


def resize_videos_and_save(dataset_path, output_path, target_size=(640, 640)):
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists. Skipping resizing.")
        return

    os.makedirs(output_path, exist_ok=True)

    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)

        # Check if it is a directory
        if os.path.isdir(class_path):
            # Create class folder in output path
            output_class_path = os.path.join(output_path, class_folder)
            os.makedirs(output_class_path, exist_ok=True)

            for video in os.listdir(class_path):
                video_path = os.path.join(class_path, video)
                video_ext = os.path.splitext(video)[1]

                if os.path.isfile(video_path) and video_ext in [".mp4", ".avi", ".mov"]:
                    try:
                        clip = VideoFileClip(video_path)
                        resized_clip = clip.resize(target_size)
                        new_video_name = f"resized_{video}"
                        new_video_path = os.path.join(output_class_path, new_video_name)
                        resized_clip.write_videofile(new_video_path, codec="libx264")
                        clip.close()
                        resized_clip.close()
                        print(f"Handled video {video_path}")
                    except Exception as e:
                        raise ValueError(f"Error when handling video {video_path}: {e}")


def detect_pose_video(pose_tracker: PoseTracker, video_path, num_landmarks_pose=33):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Initialize list to store results for each frame
    landmark_data = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image with MediaPipe Holistic models (without face detection)
        row_data = pose_tracker.process_frame_dict(image)
        # Append the row data to the list
        landmark_data.append(row_data)
        # Move to the next frame
        frame_num += 1

    header = []
    pose_header = [f"pose_{i}" for i in range(num_landmarks_pose)]
    header.extend(pose_header)
    df = pd.DataFrame(landmark_data, columns=header)
    # Define the headers for the CSV file
    return df


def process_videos(resize_dataset_video_path, destination_path):
    X, y = [], []

    if not os.path.exists(resize_dataset_video_path):
        raise ValueError(f"Path {resize_dataset_video_path} does not exist")

    print(f"Processing videos from {os.path.abspath(resize_dataset_video_path)}")
    pose_tracker = PoseTracker()

    for class_folder in os.listdir(resize_dataset_video_path):
        dir_path = os.path.join(resize_dataset_video_path, class_folder)

        # Check if it is a directory
        if not os.path.isdir(os.path.join(destination_path, class_folder)):
            os.makedirs(os.path.join(destination_path, class_folder), exist_ok=True)

        filenames = os.listdir(dir_path)
        for filename in tqdm(filenames, desc=f"Processing {class_folder}"):
            if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = os.path.join(dir_path, filename)

                csv_filename = (
                    os.path.splitext(filename)[0] + ".csv"
                )  # Change extension to .csv
                csv_path = os.path.join(destination_path, class_folder, csv_filename)

                # Check if the CSV file already exists
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    # Drop the first and second columns
                    if "frame_num" and "timestamp" in df.columns:
                        df.drop(columns=["frame_num", "timestamp"], inplace=True)

                    for column in df.columns:
                        df[column] = df[column].apply(
                            lambda x: np.array(x[1:-1].split(",")).astype(np.float32)
                        )

                    X.append(df)
                    y.append(class_folder)
                    continue
                try:
                    df = detect_pose_video(
                        pose_tracker, video_path
                    )  # Process the video
                    X.append(df)  # Append the DataFrame to the list
                    y.append(class_folder)  # Append the class to the list
                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    raise ValueError(f"Error when processing video {video_path}: {e}")

    return X, y
