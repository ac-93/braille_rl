import os
import numpy as np
import joblib
from spinup.utils.logx import restore_tf_graph
import tensorflow as tf

from braille_rl.algos.image_utils import *


# configure gpu use and supress tensorflow warnings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
Process the observation image by cropping to a bounding box, resizing to
obs_dim and threshholding to give binary image.
"""
def process_image_observation(observation, obs_dim):

    bbox  = [110,40,510,440] # [x0, y0, x1, y1], crop from 640,480 to this shape

    # crop and resize
    observation = process_image(observation,
                                gray=False, # already grayscale
                                bbox=bbox,
                                dims=(obs_dim[0], obs_dim[1]),
                                thresh=True)

    observation = observation.squeeze()

    return observation

"""
Process discrete action into one hot vector
"""
def process_action(a, act_dim):
    one_hot_a = np.eye(act_dim)[a]
    return one_hot_a

"""
Process the reward (could apply reward clipping here if needed)
"""
def process_reward(reward):
    return reward

"""
Linear annealing of epsilon value for exploration
"""
def update_eps(current_step, min_eps=0.1, max_eps=1, max_steps=1e6):
    if current_step<=max_steps:
        eps = min_eps + (max_eps - min_eps) * (1 - current_step/max_steps)
    else:
        eps=min_eps
    return eps

"""
Load a saved rl model and return neccesary parameters for resuming training
"""
def load_model(model_dir, model_save_name):

    sess = tf.compat.v1.Session(config=tf_config)

    model = restore_tf_graph(sess=sess, fpath=os.path.join(model_dir, model_save_name))
    config = load_json_obj(os.path.join(model_dir, 'config'))

    if config['rl_params']['env_type'] == 'discrete':
        if 'sim' in config['rl_params']['platform']:
            from braille_rl.envs.sim.disc_sim_braille_env.mockKBGymEnv import mockKBGymEnv as disc_mockKBGymEnv
            env = disc_mockKBGymEnv(mode=config['rl_params']['env_mode'], max_steps=config['rl_params']['max_ep_len'])
        elif 'robot' in config['rl_params']['platform']:
            from braille_rl.envs.robot.disc_ur5_braille_env.ur5GymEnv  import UR5GymEnv    as disc_UR5GymEnv
            env = disc_UR5GymEnv(mode=config['rl_params']['env_mode'], max_steps=config['rl_params']['max_ep_len'])

    elif config['rl_params']['env_type'] == 'continuous':
        if 'sim' in config['rl_params']['platform']:
            from braille_rl.envs.sim.cont_sim_braille_env.mockKBGymEnv import mockKBGymEnv as cont_mockKBGymEnv
            env = cont_mockKBGymEnv(mode=config['rl_params']['env_mode'], max_steps=config['rl_params']['max_ep_len'])
        elif 'robot' in config['rl_params']['platform']:
            from braille_rl.envs.robot.cont_ur5_braille_env.ur5GymEnv  import UR5GymEnv    as cont_UR5GymEnv
            env = cont_UR5GymEnv(mode=config['rl_params']['env_mode'], max_steps=config['rl_params']['max_ep_len'])

    print('Config: ')
    print_sorted_dict(config)
    print('')
    print('')

    # open a file, where you stored the pickled data
    file = open(os.path.join(model_dir, 'vars' + '.pkl'), 'rb')
    saved_state = joblib.load(file)
    file.close()

    print('Resume State: ')
    print_sorted_dict(saved_state)
    print('')
    print('')

    return sess, model, config['logger_kwargs'], config['rl_params'], config['network_params'], env, saved_state


"""
Store the observations in ring buffer type array of size m
"""
class StateBuffer:
    def __init__(self,m):
        self.m = m

    def init_state(self, init_obs):
        self.current_state = np.stack([init_obs]*self.m, axis=2)
        return self.current_state

    def append_state(self, obs):
        new_state = np.concatenate( (self.current_state, obs[...,np.newaxis]), axis=2)
        self.current_state = new_state[:,:,1:]
        return self.current_state


"""
Store states for a single episode, to be used by HER when adding to the
replay buffer (discrete actions)
"""
class HERBuffer:

    def __init__(self, obs_dim, act_dim, goal_dim, size):
        self.obs1_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.obs2_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.uint8)
        self.prev_acts_buf = np.zeros([size, act_dim], dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.goal_buf = np.zeros([size, goal_dim], dtype=np.uint8)
        self.ptr = 0

    def store(self, obs, act, prev_act, rew, next_obs, done, goal):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.prev_acts_buf[self.ptr] = prev_act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.goal_buf[self.ptr] = goal
        self.ptr = (self.ptr+1)

    def sample(self, idx):
        return dict(obs1=self.obs1_buf[idx],
                    obs2=self.obs2_buf[idx],
                    acts=self.acts_buf[idx],
                    prev_acts=self.prev_acts_buf[idx],
                    rews=self.rews_buf[idx],
                    done=self.done_buf[idx],
                    goal=self.goal_buf[idx])

    def reset(self):
        self.ptr = 0


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for discrete actions.
    """

    def __init__(self, obs_dim, act_dim, goal_dim, size):
        self.obs_dim = obs_dim
        self.obs1_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.obs2_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.uint8)
        self.prev_acts_buf = np.zeros([size, act_dim], dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.goal_buf = np.zeros([size, goal_dim], dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, prev_act, rew, next_obs, done, goal):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.prev_acts_buf[self.ptr] = prev_act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.goal_buf[self.ptr] = goal
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        # normalize the images before returning
        norm_obs1 = self.obs1_buf[idxs].astype('float32') / 255.
        norm_obs2 = self.obs2_buf[idxs].astype('float32') / 255.

        return dict(obs1=norm_obs1,
                    obs2=norm_obs2,
                    acts=self.acts_buf[idxs],
                    prev_acts=self.prev_acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    goal=self.goal_buf[idxs])


"""
Store states for a single episode, to be used by HER when adding to the
replay buffer (continuous actions)
"""
class ContHERBuffer:

    def __init__(self, obs_dim, act_dim, goal_dim, size):
        self.obs1_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.obs2_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.prev_acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.goal_buf = np.zeros([size, goal_dim], dtype=np.uint8)
        self.ptr = 0

    def store(self, obs, act, prev_act, rew, next_obs, done, goal):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.prev_acts_buf[self.ptr] = prev_act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.goal_buf[self.ptr] = goal
        self.ptr = (self.ptr+1)

    def sample(self, idx):
        return dict(obs1=self.obs1_buf[idx],
                    obs2=self.obs2_buf[idx],
                    acts=self.acts_buf[idx],
                    prev_acts=self.prev_acts_buf[idx],
                    rews=self.rews_buf[idx],
                    done=self.done_buf[idx],
                    goal=self.goal_buf[idx])

    def reset(self):
        self.ptr = 0


class ContReplayBuffer:
    """
    A simple FIFO experience replay buffer for continuous actions.
    """

    def __init__(self, obs_dim, act_dim, goal_dim, size):
        self.obs_dim = obs_dim
        self.obs1_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.obs2_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.prev_acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.goal_buf = np.zeros([size, goal_dim], dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, prev_act, rew, next_obs, done, goal):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.prev_acts_buf[self.ptr] = prev_act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.goal_buf[self.ptr] = goal
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        # normalize the images before returning
        norm_obs1 = self.obs1_buf[idxs].astype('float32') / 255.
        norm_obs2 = self.obs2_buf[idxs].astype('float32') / 255.

        return dict(obs1=norm_obs1,
                    obs2=norm_obs2,
                    acts=self.acts_buf[idxs],
                    prev_acts=self.prev_acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    goal=self.goal_buf[idxs])
