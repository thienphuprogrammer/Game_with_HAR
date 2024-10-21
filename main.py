from Core import Core

# from src.core_model import CoreModel
# from src.data_pipeline.preprocessing.pose_tracker import *
#
# label_list = {"Jump": 0, "Kick": 1, "Punch": 2, "Left": 3, "Right": 4, "Stand": 5}
#
#
# class TrackingSystem:
#     def __init__(self, model_path, max_stored_clips=6, save_interval=0.05):
#         self.pose_tracker = PoseTracker()
#         self.clip_storage = []
#         self.max_stored_clips = max_stored_clips
#         self.save_interval = save_interval
#         self.count_frames = 0
#         self.last_save_time = time.time()
#         self.core_model = CoreModel(model_path=model_path)
#         self.results = None
#
#     def update_and_predict_clip(self, clip):
#         results = self.results
#         clip = np.array(clip)
#         self.clip_storage.append(clip)
#         self.count_frames += len(clip)
#
#         if len(self.clip_storage) >= self.max_stored_clips:
#             # flatten the array result
#             flatten_result = []
#             for clip in self.clip_storage:
#                 for frame in clip:
#                     flatten_result.append(frame)
#             flatten_result = np.array(flatten_result)
#             results = self.core_model.predict(flatten_result, max_dim=10)
#             print(f"Prediction: {results}")
#             Max = 0
#             for i in range(len(results[0])):
#                 if results[0][i] > results[0][Max]:
#                     Max = i
#             # print the label of the result where has a value of label_list = Max
#             for key, value in label_list.items():
#                 if value == Max:
#                     results = key
#             for clip in self.clip_storage[: int(self.max_stored_clips / 1.5)]:
#                 self.count_frames -= len(clip)
#             # pop 50% of the stored clips
#             self.clip_storage = self.clip_storage[int(self.max_stored_clips / 1.5):]
#         return results
#
#     def start_camera(self):
#         # cap = cv2.VideoCapture("./data/raw/HAR/Train/Punch/Punch_10.mp4")
#         cap = cv2.VideoCapture("./data/7752246914108791071.mp4")
#         # cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             raise IOError("Cannot open webcam")
#
#         print("Press 'q' to quit")
#
#         current_clip = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             frame = cv2.flip(frame, 1)  # Mirror the frame
#             row_data = self.pose_tracker.process_frame(frame)
#             current_clip.append(row_data)
#
#             if time.time() - self.last_save_time >= self.save_interval:
#                 # self.core_model.predict(current_clip)
#                 self.results = self.update_and_predict_clip(current_clip)
#                 current_clip = []
#                 self.last_save_time = time.time()
#
#             # Display the FPS
#             fps = self.pose_tracker.get_fps()
#             if self.results is not None:
#                 cv2.putText(
#                     frame,
#                     f"Prediction: {self.results}",
#                     (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (0, 255, 0),
#                     2,
#                 )
#             cv2.putText(
#                 frame,
#                 f"FPS: {fps:.2f}",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 255, 0),
#                 2,
#             )
#             cv2.putText(
#                 frame,
#                 f"Clips stored: {self.count_frames}",
#                 (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 255, 0),
#                 2,
#             )
#
#             # Display the frame
#             cv2.imshow("Webcam", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#
#         cap.release()
#         cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     pose_tracker = TrackingSystem(model_path="./results/models/CNN_LSTM.h5")
#     pose_tracker.start_camera()
#     del pose_tracker

if __name__ == '__main__':
    oCore = Core(model_path="./results/models/models.h5")
    oCore.main_loop()
