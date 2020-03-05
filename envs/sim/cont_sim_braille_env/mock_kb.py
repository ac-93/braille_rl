import os
import numpy as np
import pandas as pd
from envs.sim.image_utils import load_video_frames, load_json_obj, process_image, print_sorted_dict

class mockKB:

    def __init__(self, mode='arrows'):

        self.sensor_angle = -145
        self.constant_tap_depth = 2            # done every move
        self.tap_move     = [0, 0, 4, 0, 0, 0] # done on randomise pos
        self.mode = mode
        self.img_width, self.img_height, self.img_channels = 640, 480, 1

        # arrows setup
        if 'arrows' in self.mode:
            self.work_frame = [-126.5, -416.5, 28.75, -180, 0, -90] # 28.5
            self.relative_EE_pos = [0,0,0,0,0,self.sensor_angle] # orientate sensor so braille is upright in camera

            # limit the regions the arm can move relative to workframe for safety purposes
            self.x_min, self.x_max = -20, 0
            self.y_min, self.y_max = -20, 20
            self.z_min, self.z_max = 0, 8

            # max and min action predicted by the NN (have to be equal for SAC)
            self.min_action  = -10
            self.max_action  = 10
            self.z_max_depth = 6 # z gets mapped to [0, z_max_depth]

            self.training_data_dir = os.path.join( os.path.dirname(__file__), '../../../data/grid_data/arrows')
            self.label_df = pd.read_csv(os.path.join(self.training_data_dir, 'targets_video.csv'))
            self.label_list = self.label_df['obj_lbl'].values
            self.label_arr = np.array([i.split('_') for i in self.label_list]).astype(np.float)
            self.label_arr[:,[0,1]] = self.label_arr[:,[1,0]]  # switch x,y as difference between collected data and env

        # alphabet
        elif 'alphabet' in self.mode:
            self.work_frame = [21, -445, 28.75, -180, 0, -90] # mid keyboard origin
            self.relative_EE_pos = [0,0,0,0,0,self.sensor_angle] # orientate sensor so braille is upright in camera

            # limit the regions the arm can move relative to workframe for safety purposes
            self.x_min, self.x_max = -30, 30
            self.y_min, self.y_max = -90, 90
            self.z_min, self.z_max = 0, 8

            # max and min action predicted by the NN (have to be equal for SAC)
            self.min_action  = -20
            self.max_action  = 20
            self.z_max_depth = 6 # z gets mapped from [min_action, max_action] to [0, z_max_depth]

            # using training data to gather image from collected set
            self.training_data_dir = os.path.join( os.path.dirname(__file__), '../../../data/grid_data/alphabet')
            self.label_df = pd.read_csv(os.path.join(self.training_data_dir, 'targets_video.csv'))
            self.label_list = self.label_df['obj_lbl'].values
            self.label_arr = np.array([i.split('_') for i in self.label_list]).astype(np.float)
            self.label_arr[:,[0,1]] = self.label_arr[:,[1,0]]  # switch x,y as difference between collected data and env
        else:
            sys.exit('Incorrect mode specified, please choose from [\'arrows\', \'alphabet\']')


    def reset(self):
        obs, button_pressed = self.randomise_start_pos()
        return obs, button_pressed

    def randomise_start_pos(self):

        # change EE position
        self.relative_EE_pos[0] = np.random.uniform(low=self.x_min, high=self.x_max)
        self.relative_EE_pos[1] = np.random.uniform(low=self.y_min, high=self.y_max)
        self.relative_EE_pos[2] = 0 # fixed z height

        obs, button_pressed = self.drop_move(self.tap_move)

        return obs, button_pressed

    def reset_pos(self, ee_pos):
        self.relative_EE_pos = ee_pos

    def drop_move(self, move):
        drop_pos = [sum(x) for x in zip(self.relative_EE_pos, move)]
        drop_pos = self.check_EE_lims(drop_pos)
        obs, button_pressed = self.get_observation(drop_pos)
        return obs, button_pressed

    def check_EE_lims(self, EE_pos):

        if (EE_pos[0]<self.x_min): EE_pos[0]=self.x_min
        if (EE_pos[0]>self.x_max): EE_pos[0]=self.x_max

        if (EE_pos[1]<self.y_min): EE_pos[1]=self.y_min
        if (EE_pos[1]>self.y_max): EE_pos[1]=self.y_max

        if (EE_pos[2]<self.z_min): EE_pos[2]=self.z_min
        if (EE_pos[2]>self.z_max): EE_pos[2]=self.z_max

        return EE_pos

    def apply_action(self, actions):

        dx = actions[0]
        dy = actions[1]
        dz = actions[2]

        # normalize to [0,max_depth]mm range
        # required because networks usually predict [-1,1] range
        # which is scaled to max_action
        dz = self.z_max_depth * ( (dz-(self.min_action))/((self.max_action)-(self.min_action)) )

        # change EE position (x and y)
        self.relative_EE_pos[0] = self.relative_EE_pos[0] + dx
        self.relative_EE_pos[1] = self.relative_EE_pos[1] + dy

        # additional 2mm tapping depth on each move now [2,7]mm range
        move = [0, 0, self.constant_tap_depth+dz, 0, 0, 0]
        obs, button_pressed  = self.drop_move(move)

        return obs, button_pressed


    def get_observation(self, pos):

        # find closest saved image to ee_pos
        curr_pos = pos[0:3]
        idx = (np.abs(self.label_arr - curr_pos)).argmin()
        distances = np.sum(np.power(self.label_arr - curr_pos, 2), axis=1)
        idx = distances.argmin()

        # randomly pull an image from the training dataset that matches this idx
        video_filename = self.label_df.iloc[idx]['sensor_video']
        video_path = os.path.join(self.training_data_dir, 'videos', video_filename)
        video = load_video_frames(video_path)
        obs_img = video[1,...]

        # EDIT don't do processing here to match that of the real env
        obs_img = process_image(obs_img)
        obs_img = obs_img.squeeze()

        # get the button pressed during data collection
        button_pressed = self.label_df.iloc[idx]['pressed_buttons']

        return obs_img, button_pressed

    def get_action_dimension(self):
        return 3 # x, y, td

    def get_observation_dimension(self):
        # return [self.img_width, self.img_height, self.img_channels]
        return [self.img_width, self.img_height, self.img_channels]
