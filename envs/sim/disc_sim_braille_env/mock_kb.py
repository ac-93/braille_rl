import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import cv2
from algos.image_utils import load_video_frames, process_image

class mockKB:

    def __init__(self, mode='arrows'):

        self.mode = mode
        self.img_width, self.img_height, self.img_channels = 640, 480, 1

        # arrows setup
        if 'arrows' in self.mode:
            self.n_x_steps = 2
            self.n_y_steps = 3

            self.obs_list = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
            self.pos_arr = [ ['NONE', 'UP', 'NONE'],
                             ['LEFT', 'DOWN', 'RIGHT']]

            # using training data to gather image from collected set
            self.training_data_dir = os.path.join( os.path.dirname(__file__), '../../../data/supervised_data/arrows/train')
            self.label_df = pd.read_csv(os.path.join(self.training_data_dir, 'targets_video.csv'))

        # alphabet
        elif 'alphabet' in self.mode:
            self.n_x_steps = 4
            self.n_y_steps = 10
            self.obs_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                             'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                             'SPACE', 'NONE']

            self.pos_arr = [ ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
                             ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'NONE'],
                             ['NONE',  'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'NONE', 'NONE'],
                             ['NONE', 'NONE','SPACE', 'SPACE', 'SPACE', 'SPACE', 'SPACE', 'SPACE', 'NONE', 'NONE']]

            # using training data to gather image from collected set
            self.training_data_dir = os.path.join( os.path.dirname(__file__), '../../../data/supervised_data/alphabet/train')
            self.label_df = pd.read_csv(os.path.join(self.training_data_dir, 'targets_video.csv'))
        else:
            sys.exit('Incorrect mode specified, please choose from [\'arrows\', \'alphabet\']')

        self.x_idx, self.y_idx = 0, 0

        self.action_list = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PRESS']



    def reset(self):
        obs, btn_pressed = self.randomise_start_pos()
        # obs, btn_pressed = self.get_observation('tap')
        return obs, btn_pressed

    def randomise_start_pos(self):
        self.x_idx = np.random.randint(self.n_x_steps)
        self.y_idx = np.random.randint(self.n_y_steps)
        self.x_idx, self.y_idx = self.check_idx_lims(self.x_idx, self.y_idx)

        obs, btn_pressed = self.get_observation('tap')

        return obs, btn_pressed

    def reset_pos(self, x_idx, y_idx):
        self.x_idx = x_idx
        self.y_idx = y_idx

    def check_idx_lims(self, x_idx, y_idx):
        if x_idx < 0: x_idx = 0
        if y_idx < 0: y_idx = 0
        if x_idx > self.n_x_steps-1: x_idx = self.n_x_steps-1
        if y_idx > self.n_y_steps-1: y_idx = self.n_y_steps-1
        return x_idx, y_idx

    def apply_action(self, act_idx):

        action = self.action_list[act_idx]

        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            if action == 'UP':
                self.x_idx -= 1
            elif action == 'DOWN':
                self.x_idx += 1
            elif action == 'LEFT':
                self.y_idx -= 1
            elif action == 'RIGHT':
                self.y_idx += 1

            self.x_idx, self.y_idx = self.check_idx_lims(self.x_idx, self.y_idx)

            obs, btn_pressed = self.get_observation('tap')

        else:
            if action == 'TAP':
                obs, btn_pressed = self.get_observation('tap')
            if action == 'PRESS':
                obs, btn_pressed = self.get_observation('press')

        return obs, btn_pressed


    def get_observation(self, move):
        # if tap move get the value of the key that its above
        if move == 'noop':
            obs = 'NONE'
            button_pressed = None

        if move == 'tap':
            obs = self.pos_arr[self.x_idx][self.y_idx]
            button_pressed = None

        if move == 'press':
            obs = self.pos_arr[self.x_idx][self.y_idx]
            if obs == 'NONE':
                button_pressed = None
            else:
                button_pressed = obs

        # randomly pull an image from the training dataset that matches this label
        video_filename = self.label_df.loc[self.label_df['obj_lbl'] == obs].sample()['sensor_video'].values[0]
        video_path = os.path.join(self.training_data_dir, 'videos', video_filename)
        video = load_video_frames(video_path)
        obs_img = video[1,...]

        # EDIT don't do much processing here to match that of the real env where only
        # a raw tactile image is returned
        obs_img = process_image(obs_img)
        obs_img = obs_img.squeeze()

        return obs_img, button_pressed

    def get_action_dimension(self):
        return len(self.action_list)

    def get_observation_dimension(self):
        # return [self.img_width, self.img_height, self.img_channels]
        return [self.img_width, self.img_height, self.img_channels]
