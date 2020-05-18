import sys
import numpy as np
import copy
import gym
import braille_rl.envs.sim.disc_sim_braille_env.mock_kb as mkb

class mockKBGymEnv(gym.Env):

    def __init__(self,
                 mode='arrows',
                 max_steps=100):

        if mode not in ['arrows', 'alphabet']:
            sys.exit('Incorrect mode specified, please choose from [\'arrows\',  \'alphabet\']')

        # initialise variables
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
        self.MKB = mkb.mockKB(mode=mode)

        # setup observation and action spaces (z action may be smaller)
        self.action_list = self.MKB.action_list
        self.n_actions = len(self.action_list)
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # image dimensions for sensor
        self.obs_dim = self.MKB.get_observation_dimension()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.obs_dim, dtype=np.uint8)

        # rewards defined here for HER
        self.min_rew  = 0
        self.max_rew  = 1
        self.step_rew = 0

        self.seed()


    def reset(self):

        self.terminated = 0
        self._env_step_counter = 0
        self.latest_button = None

        # move robot to home position, random position, leave here etc
        self._observation, self.latest_button = self.MKB.reset()

        return self._observation

    def get_env_state(self):
        return copy.deepcopy([self.terminated,
                self._env_step_counter,
                self.latest_button,
                self.goal_button,
                self._observation,
                self.MKB.x_idx,
                self.MKB.y_idx])

    def set_env_state(self, env_state):
        # set all env params for resuming correctly
        self.terminated        = env_state[0]
        self._env_step_counter = env_state[1]
        self.latest_button     = env_state[2]
        self.goal_button       = env_state[3]
        self._observation      = env_state[4]

        # set the ur5 position to where training stopped
        self.MKB.reset_pos(x_idx=env_state[5], y_idx=env_state[6])

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self._observation, self.latest_button = self.MKB.apply_action(action)
        self._env_step_counter += 1

        reward = self._reward()
        done   = self._termination()

        return self._observation, reward, done, {}


    def _termination(self):
        if (self.terminated or self._env_step_counter>=self._max_steps):
            return True
        return False

    def _reward(self):

        # button has been pressed
        if self.latest_button is not None:
            # correct button pressed
            if self.latest_button == self.goal_button:
                # print('Goal Reached')
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
