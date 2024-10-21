from os import environ

import pygame as pg
from pygame.locals import *

from Const import *
from Map import Map
from MenuManager import MenuManager
from Sound import Sound

from src.core_model import CoreModel
from src.data_pipeline.preprocessing.pose_tracker import *

label_list = {"Jump": 0, "Kick": 1, "Punch": 2, "Left": 3, "Right": 4, "Stand": 5}


class Core(object):
    def __init__(self, model_path, max_stored_clips=6, save_interval=0.05):
        environ['SDL_VIDEO_CENTERED'] = '1'
        pg.mixer.pre_init(44100, -16, 2, 1024)
        pg.init()
        pg.display.set_caption('mario Game')
        pg.display.set_mode((WINDOW_W, WINDOW_H))

        self.screen = pg.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pg.time.Clock()

        self.oWorld = Map('1-1')
        self.oSound = Sound()
        self.oMM = MenuManager(self)

        self.run = True
        self.keyR = False
        self.keyL = False
        self.keyU = False
        self.keyD = False
        self.keyShift = False

        self.pose_tracker = PoseTracker()
        self.clip_storage = []
        self.max_stored_clips = max_stored_clips
        self.save_interval = save_interval
        self.count_frames = 0
        self.last_save_time = time.time()
        self.core_model = CoreModel(model_path=model_path)
        self.results = None

    def update_and_predict_clip(self, clip):
        results = self.results
        clip = np.array(clip)
        self.clip_storage.append(clip)
        self.count_frames += len(clip)

        if len(self.clip_storage) >= self.max_stored_clips:
            # flatten the array result
            flatten_result = []
            for clip in self.clip_storage:
                for frame in clip:
                    flatten_result.append(frame)
            flatten_result = np.array(flatten_result)
            results = self.core_model.predict(flatten_result, max_dim=10)
            print(f"Prediction: {results}")
            Max = 0
            for i in range(len(results[0])):
                if results[0][i] > results[0][Max]:
                    Max = i
            # print the label of the result where has a value of label_list = Max
            for key, value in label_list.items():
                if value == Max:
                    results = key
            for clip in self.clip_storage[: int(self.max_stored_clips / 1.5)]:
                self.count_frames -= len(clip)
            # pop 50% of the stored clips
            self.clip_storage = self.clip_storage[int(self.max_stored_clips / 1.5):]
        return results


    def main_loop(self):
        # cap = cv2.VideoCapture("./data/raw/HAR/Train/Punch/Punch_10.mp4")
        cap = cv2.VideoCapture("./data/7752246914108791071.mp4")
        # cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        print("Press 'q' to quit")
        current_clip = []

        while self.run:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # Mirror the frame
            row_data = self.pose_tracker.process_frame(frame)
            current_clip.append(row_data)

            self.input()
            self.update()
            self.render()
            self.clock.tick(FPS)

            if time.time() - self.last_save_time >= self.save_interval:
                self.results = self.update_and_predict_clip(current_clip)
                current_clip = []
                self.last_save_time = time.time()

            if self.results is not None:
                cv2.putText(
                    frame,
                    f"Prediction: {self.results}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # Display the frame
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def input(self):
        if self.get_mm().currentGameState == 'Game':
            self.input_player()
        else:
            self.input_menu()

    def input_player(self):

        if self.results == "Jump":
            self.keyU = True
        elif self.results == "Kick":
            self.keyD = True
        elif self.results == "Right":
            self.keyR = True
            self.keyL = False
        elif self.results == "Left":
            self.keyL = True
            self.keyR = False
        elif self.results == "Stand":
            self.keyL = False
            self.keyR = False
            self.keyU = False
            self.keyD = False


        for e in pg.event.get():

            if e.type == pg.QUIT:
                self.run = False

            elif e.type == KEYDOWN:
                if e.key == K_RIGHT:
                    self.keyR = True
                elif e.key == K_LEFT:
                    self.keyL = True
                elif e.key == K_DOWN:
                    self.keyD = True
                elif e.key == K_UP:
                    self.keyU = True
                elif e.key == K_LSHIFT:
                    self.keyShift = True

            elif e.type == KEYUP:
                if e.key == K_RIGHT:
                    self.keyR = False
                elif e.key == K_LEFT:
                    self.keyL = False
                elif e.key == K_DOWN:
                    self.keyD = False
                elif e.key == K_UP:
                    self.keyU = False
                elif e.key == K_LSHIFT:
                    self.keyShift = False


    def input_menu(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.run = False

            elif e.type == KEYDOWN:
                if e.key == K_RETURN:
                    self.get_mm().start_loading()

    def update(self):
        self.get_mm().update(self)

    def render(self):
        self.get_mm().render(self)

    def get_map(self):
        return self.oWorld

    def get_mm(self):
        return self.oMM

    def get_sound(self):
        return self.oSound
