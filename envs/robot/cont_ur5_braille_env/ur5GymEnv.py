import sys
import gym
import numpy as np
import copy

import evdev
import threading
from select import select

import envs.robot.cont_ur5_braille_env.ur5_with_tactip as ur5

class UR5GymEnv(gym.Env):

    def __init__(self,
                 mode='arrows',
                 max_steps=100):

        if mode not in ['arrows', 'alphabet']:
            sys.exit('Incorrect mode specified, please choose from [\'arrows\', \'alphabet\']')

        # initialise variables
        self._observation = []
        self._env_step_counter = 0
        self._max_steps = max_steps
        self.terminated = 0

        # reward variables
        self.goal_button = None
        self.latest_button = None

        if 'arrows' in mode:
            self.goal_list = ['UP','DOWN','LEFT','RIGHT']
        elif 'alphabet' in mode:
            self.goal_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                              'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                              'SPACE']

        # initialse ur5, go to home pos
        self._UR5 = ur5.UR5(mode=mode)

        # setup observation and action spaces (z action is scaled down by factor of 10)
        self.action_space = gym.spaces.Box(low=self._UR5.min_action, high=self._UR5.max_action, shape=(self._UR5.get_action_dimension(),), dtype=np.float32)

        # image dimensions for sensor
        self.obs_dim = self._UR5.get_observation_dimension()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.obs_dim, dtype=np.uint8)
        self._observation = self._UR5.get_observation()

        # create and start a monitor for keyboard
        # make sure correct keyboard
        devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
        for dev in devices:
            if 'HID 04b4:6001' in dev.name:
                keyboard_path = dev.fn

        keyboard = evdev.InputDevice(keyboard_path)
        print('Keyboard Device: ', keyboard.path, keyboard.name)

        self.keypress_thread = threading.Thread(target=self.monitor_keyboard, args=(keyboard,))
        self.keypress_thread.daemon = True  # Daemonize thread
        self.stop_event = threading.Event()
        self.keypress_thread.start() # Start the execution

        # rewards defined here for HER
        self.min_rew  = 0
        self.max_rew  = 1
        self.step_rew = 0

        self.seed()

    def close(self):
        self.stop_event.set()
        self.keypress_thread.join()
        self.keypress_thread = None
        print('Closed Keyboard Thread')

        self._UR5.close()

    def reset(self):

        self.terminated = 0
        self._env_step_counter = 0
        self.latest_button = None

        self._observation = self._UR5.reset()

        return self._observation

    def get_env_state(self):
        return copy.deepcopy([self.terminated,
                self._env_step_counter,
                self.latest_button,
                self.goal_button,
                self._observation,
                self._UR5.relative_EE_pos])

    def set_env_state(self, env_state):
        # set all env params for resuming correctly
        self.terminated           = env_state[0]
        self._env_step_counter    = env_state[1]
        self.latest_button        = env_state[2]
        self.goal_button          = env_state[3]
        self._observation         = env_state[4]

        # set the ur5 position to where training stopped
        self._UR5.reset_pos(ee_pos=env_state[5])

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self._observation = self._UR5.apply_action(action)
        self._env_step_counter += 1

        reward = self._reward()
        done = self._termination()

        return self._observation, reward, done, {}


    def _termination(self):
        if (self.terminated or self._env_step_counter>=self._max_steps):
            return True
        return False

    def monitor_keyboard(self, dev):
        with dev.grab_context(): # grab exclusive access to keyboard
            while not self.stop_event.isSet(): # break loop when needed
                select([dev], [], [], 0.25) # split readloop into select and read to include timeout and not block
                try:
                    for event in dev.read():
                        if event.type == evdev.ecodes.EV_KEY:
                            cat_event = evdev.categorize(event)
                            if cat_event.keystate == cat_event.key_down: # on press
                                key_str = cat_event.keycode.split('_', 1)[1] # get only the letter
                                if key_str in self.goal_list:
                                    self.latest_button = key_str
                                else:
                                    self.latest_button = 'not_in_goal_list'
                except BlockingIOError:
                    pass

    def _reward(self):

        # button has been pressed
        if self.latest_button is not None:
            # correct button pressed
            if self.latest_button == self.goal_button:
                print('Goal Reached')
                reward = self.max_rew
                self.terminated = 1
            # incorrect button pressed
            else:
                # print('Incorrect Button Press')
                reward = self.min_rew
                self.terminated = 1
        else:
            reward = self.step_rew

        return reward
