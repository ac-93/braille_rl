import numpy as np

from cri.robot import SyncRobot, AsyncRobot
from cri.controller import RTDEController

from vsp.v4l2_camera import V4l2VideoCamera
from vsp.video_stream import CvVideoDisplay, CvVideoOutputFile
from vsp.processor import CameraStreamProcessorMT, AsyncProcessor

def make_robot():
    return AsyncRobot(SyncRobot(RTDEController(ip='127.0.0.1')))  # local host sim
    # return AsyncRobot(SyncRobot(RTDEController(ip='192.11.72.10'))) # real robot

def make_sensor():
    return AsyncProcessor(CameraStreamProcessorMT(
            camera=V4l2VideoCamera(device_path="/dev/video4", # depends on cameras connected to pc
                                   frame_size=(640, 480),
                                   fourcc = 'MP4V',
                                   num_buffers=1,
                                   is_color=False),
            display=CvVideoDisplay(name='Raw Sensor Image'),
            writer=CvVideoOutputFile(),
        ))


class UR5:

    def __init__(self, mode='arrows'):

        # robot setup
        self.robot_tcp    = [0, 0, 125.0, 0, 0, 0]             # tactip adds 125mm to tcp
        self.base_frame   = [0, 0, 0, 0, 0, 0]                 # origin of ur5 base
        self.home_pose    = [109.1, -487.0, 320, -180, 0, -90] # safe position of ur5
        self.sensor_angle = -145                               # angle for uprigth braille
        self.tap_move     = [0, 0, 5, 0, 0, 0]                 # depth for a tap action that doesnt activate the key
        self.press_move   = [0, 0, 8, 0, 0, 0]                 # depth for press action that activates the key

        self.mode = mode

        # arrows setup
        if 'arrows' in self.mode:
            self.work_frame = [-126.5, -416.5, 28.75, -180, 0, -90] # DOWN arrow origin
            self.relative_EE_pos = [0,0,0,0,0,self.sensor_angle] # orientate sensor so braille is upright in camera

            self.x_min, self.x_max = -19, 0
            self.y_min, self.y_max = -19, 19
            self.n_x_steps = 2
            self.n_y_steps = 3

            # ['BLANK', 'UP',  'BLANK'],
            # ['LEFT', 'DOWN', 'RIGHT'],
            self.x_grid = np.array([ [-19, -19, -19],
                                     [ 0,   0,   0 ] ])

            self.y_grid = np.array([ [-19,  0,   19],
                                     [-19,  0,   19]  ])

        # alphabet
        elif 'alphabet' in self.mode:
            self.work_frame = [111, -473.5, 28.75, -180, 0, -90] # Q button origin
            self.relative_EE_pos = [0,0,0,0,0,self.sensor_angle] # orientate sensor so braille is upright in camera

            self.x_min, self.x_max = 0, 57
            self.y_min, self.y_max = -185, 0
            self.n_x_steps = 4
            self.n_y_steps = 10

            # ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            # ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'BLANK'],
            # ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BLANK', 'BLANK', 'BLANK'],
            # ['BLANK', 'BLANK', 'SPACE', 'SPACE', 'SPACE', 'SPACE', 'SPACE', 'SPACE', 'BLANK', 'BLANK']
            self.x_grid = np.array([ [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                                     [19, 19, 19, 19, 19, 19, 19, 19, 19, 19],
                                     [38, 38, 38, 38, 38, 38, 38, 38, 38, 38],
                                     [57, 57, 57, 57, 57, 57, 57, 57, 57, 57] ])

            self.y_grid = np.array([ [0,   -19, -38, -57, -76, -95,  -114, -133, -152, -171],
                                     [-5,  -24, -43, -62, -81, -100, -119, -138, -157, -176],
                                     [-14, -33, -52, -71, -90, -109, -128, -147, -166, -185],
                                     [-5,  -24, -43, -62, -81, -100, -119, -138, -160, -180] ])

        else:
            sys.exit('Incorrect mode specified, please choose from [\'arrows\', \'alphabet\']')

        self.x_idx, self.y_idx = 0, 0

        self.action_list = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PRESS']

        # make the robot
        self.robot = make_robot()

        # configure robot
        self.robot.tcp = self.robot_tcp
        self.robot.linear_speed = 30

        # move to home position
        print("Moving to home position ...")
        self.robot.coord_frame = self.base_frame
        self.robot.move_linear(self.home_pose)

        # move to origin of work frame
        print("Moving to work frame origin ...")
        self.robot.coord_frame = self.work_frame
        self.robot.move_linear(self.relative_EE_pos)

        self.robot.linear_speed = 100 # change to higher
        # self.robot.linear_acceleration = 1000

        # make the sensor
        self.img_width, self.img_height, self.img_channels = 640, 480, 1 # rescale to this size
        self.sensor = make_sensor()
        self.sensor.process(num_frames=2) # start the buffer

    def reset(self):
        # print('Resetting with random start position.')
        obs = self.randomise_start_pos()
        # obs = self.drop_move(self.tap_move)
        return obs

    def close(self):
        if self.robot is not None:
            # move to home position
            print("Moving to home position ...")
            self.robot.coord_frame = self.base_frame
            self.robot.move_linear(self.home_pose)

            self.robot.close()
            self.robot = None
            print('Robot Closed')

        if self.sensor is not None:
            self.sensor.close()
            self.sensor = None
            print('Sensor Closed')

    def randomise_start_pos(self):
        self.x_idx = np.random.randint(self.n_x_steps)
        self.y_idx = np.random.randint(self.n_y_steps)

        self.x_idx, self.y_idx = self.check_idx_lims(self.x_idx, self.y_idx)

        # change EE position
        self.relative_EE_pos[0] = self.x_grid[self.x_idx, self.y_idx]
        self.relative_EE_pos[1] = self.y_grid[self.x_idx, self.y_idx]
        self.relative_EE_pos[2] = 0 # fixed z height

        self.move_safely()

        obs = self.drop_move(self.tap_move)

        return obs

    def reset_pos(self, ee_pos, x_idx, y_idx):
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.relative_EE_pos = ee_pos
        self.move_safely()

    def check_idx_lims(self, x_idx, y_idx):
        if x_idx < 0: x_idx = 0
        if y_idx < 0: y_idx = 0
        if x_idx > self.n_x_steps-1: x_idx = self.n_x_steps-1
        if y_idx > self.n_y_steps-1: y_idx = self.n_y_steps-1

        return x_idx, y_idx

    def check_EE_lims(self, EE_pos):

        if (EE_pos[0]<self.x_min): EE_pos[0]=self.x_min
        if (EE_pos[0]>self.x_max): EE_pos[0]=self.x_max

        if (EE_pos[1]<self.y_min): EE_pos[1]=self.y_min
        if (EE_pos[1]>self.y_max): EE_pos[1]=self.y_max

        return EE_pos

    def apply_action(self, act_idx):

        action = self.action_list[act_idx]

        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            if action == 'UP':
                self.x_idx -= 1
            elif action == 'DOWN':
                self.x_idx += 1
            elif action == 'LEFT':
                self.y_idx += 1
            elif action == 'RIGHT':
                self.y_idx -= 1

            self.x_idx, self.y_idx = self.check_idx_lims(self.x_idx, self.y_idx)

            # change EE position
            self.relative_EE_pos[0] = self.x_grid[self.x_idx, self.y_idx]
            self.relative_EE_pos[1] = self.y_grid[self.x_idx, self.y_idx]

            self.move_safely()

            obs = self.drop_move(self.tap_move)

        else:
            if action == 'TAP':
                move = self.tap_move
            if action == 'PRESS':
                move = self.press_move

            obs = self.drop_move(move)

        return obs

    def drop_move(self, move):
        # new positon from summing two lists
        drop_pos = [sum(x) for x in zip(self.relative_EE_pos, move)]

        self.robot.move_linear(drop_pos)             # move down
        obs = self.get_observation()                 # get image
        self.robot.move_linear(self.relative_EE_pos) # move up
        return obs

    def move_safely(self):
        # limit the regions the arm can move for safety purposes
        self.x_idx, self.y_idx = self.check_idx_lims(self.x_idx, self.y_idx)
        self.relative_EE_pos = self.check_EE_lims(self.relative_EE_pos)

        # move to new end effector pos
        self.robot.move_linear(self.relative_EE_pos)

    def get_observation(self):
        # pull 2 frames from buffer (of size 1) and use second frame
        # this ensures we are not one step delayed
        frames = self.sensor.process(num_frames=2)
        observation = frames[1]
        return observation

    def get_action_dimension(self):
        # up, down, left, right, press
        dim = len(self.action_list)
        return dim

    def get_observation_dimension(self):
        return [self.img_width, self.img_height, self.img_channels]
