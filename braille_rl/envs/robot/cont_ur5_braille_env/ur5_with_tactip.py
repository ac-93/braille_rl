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
            camera=V4l2VideoCamera(device_path="/dev/video4",
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
        self.sensor_angle = -145                               # angle for upright braille
        self.constant_tap_depth = 2                            # tap depth done every move
        self.tap_move     = [0, 0, 4, 0, 0, 0]                 # used when episode is reset
        self.mode = mode

        # arrows setup
        if 'arrows' in self.mode:

            self.work_frame = [-126.5, -416.5, 29.5, -180, 0, -90] # center of arrows
            self.relative_EE_pos = [0,0,0,0,0,self.sensor_angle]   # orientate sensor so braille is upright in camera

            # limit the regions the arm can move relative to workframe for safety purposes
            self.x_min, self.x_max = -20, 0
            self.y_min, self.y_max = -20, 20
            self.z_min, self.z_max = 0, 8

            # max and min action predicted by the NN (have to be equal for SAC)
            self.min_action  = -10
            self.max_action  = 10
            self.z_max_depth = 6 # z gets mapped from [min_action, max_action] to [constant_tap_depth, z_max_depth+constant_tap_depth]

        # alphabet setup
        elif 'alphabet' in self.mode:

            self.work_frame = [21, -445, 28.75, -180, 0, -90] # center of keyboard
            self.relative_EE_pos = [0,0,0,0,0,self.sensor_angle] # orientate sensor so braille is upright in camera

            # limit the regions the arm can move relative to workframe for safety purposes
            self.x_min, self.x_max = -30, 30
            self.y_min, self.y_max = -90, 90
            self.z_min, self.z_max = 0, 8

            # max and min action predicted by the NN (have to be equal for SAC)
            self.min_action  = -20
            self.max_action  = 20
            self.z_max_depth = 6 # z gets mapped from [min_action, max_action] to [constant_tap_depth, z_max_depth+constant_tap_depth]

        else:
            sys.exit('Incorrect mode specified, please choose from [\'arrows\', \'alphabet\']')

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

        # make the sensor
        self.img_width, self.img_height, self.img_channels = 640, 480, 1
        self.sensor = make_sensor()
        self.sensor.process(num_frames=2)


    def reset(self):
        obs = self.randomise_start_pos()
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

        # change EE position
        self.relative_EE_pos[0] = np.random.uniform(low=self.x_min, high=self.x_max)
        self.relative_EE_pos[1] = np.random.uniform(low=self.y_min, high=self.y_max)
        self.relative_EE_pos[2] = 0 # fixed z height

        self.move_safely()

        obs = self.drop_move(self.tap_move)

        return obs

    def reset_pos(self, ee_pos):
        self.relative_EE_pos = ee_pos
        self.move_safely()

    def drop_move(self, move):
        # new positon from summing two lists
        drop_pos = [sum(x) for x in zip(self.relative_EE_pos, move)]
        drop_pos = self.check_EE_lims(drop_pos)

        self.robot.move_linear(drop_pos)             # move down
        obs = self.get_observation()                 # get image
        self.robot.move_linear(self.relative_EE_pos) # move up
        return obs

    def move_safely(self):
        # limit the regions the arm can move for safety purposes
        self.relative_EE_pos = self.check_EE_lims(self.relative_EE_pos)
        # move to new end effector pos
        self.robot.move_linear(self.relative_EE_pos)

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

        # change EE position (x and y)
        self.relative_EE_pos[0] = self.relative_EE_pos[0] + dx
        self.relative_EE_pos[1] = self.relative_EE_pos[1] + dy
        self.move_safely()

        # normalize dz to [0,6]mm range
        dz = self.z_max_depth * ( (dz-(self.min_action))/((self.max_action)-(self.min_action)) )

        # additional 2mm tapping depth on each move now [2,8]mm range
        move = [0, 0, self.constant_tap_depth+dz, 0, 0, 0]
        obs  = self.drop_move(move)

        return obs


    def get_observation(self):
        # pull 2 frames from buffer (of size 1) and use second frame
        # this ensures we are not one step delayed
        frames = self.sensor.process(num_frames=2)
        observation = frames[1]
        return observation

    def get_action_dimension(self):
        # x, y, z
        return 3

    def get_observation_dimension(self):
        return [self.img_width, self.img_height, self.img_channels]
