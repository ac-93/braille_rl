import os
from spinup.utils.run_utils import setup_logger_kwargs

from braille_rl.algos.cont_sac_algo.sac import sac

# atari
network_params = {
    'input_dims':   [100,100,1],
    'conv_filters': (32, 64, 64),
    'kernel_width': (8,  4,  3),
    'strides':      (4,  2,  1),
    'pooling':'none',
    'pooling_width':2,
    'pooling_strides':1,
    'dense_units':(1024,512),
    'hidden_activation':'relu',
    'batch_norm':False,
    'dropout':0.0
    }

rl_params = {
    # ==== env params ====
    'platform':'sim',
    # 'platform':'robot',
    'env_type':'continuous',
    'env_mode':'arrows',
    # 'env_mode':'alphabet',

    # ==== control params ====
    'seed':int(2),
    'epochs':int(100),
    'steps_per_epoch':250,
    'replay_size':int(1e5),
    'update_freq':1,
    'n_updates': 2,
    'batch_size':32,
    'start_steps':200,
    'max_ep_len':35,
    'num_tests':20,
    'save_freq':None, # Warning, uncompressed replay buffer takes a lot of space

    # ==== rl params ====
    'use_HER':True,
    'use_prev_a':True,
    'gamma':0.95,
    'polyak':0.995,
    'act_lr' :0.0005,
    'crit_lr':0.0005,
    'alph_lr':0.001,

    # ==== noise params ====
    'alpha': 'auto',         # fixed or auto balance
    'target_entropy':-6  # fixed or auto define with act_dim
}

saved_model_dir = os.path.join('../saved_models/', rl_params['platform'], rl_params['env_type'], rl_params['env_mode'])
logger_kwargs = setup_logger_kwargs(exp_name='cont_sac',
                                    seed=rl_params['seed'],
                                    data_dir=saved_model_dir,
                                    datestamp=False)

if 'sim' in rl_params['platform']:
    from braille_rl.envs.sim.cont_sim_braille_env.mockKBGymEnv import mockKBGymEnv
    env = mockKBGymEnv(mode=rl_params['env_mode'], max_steps=rl_params['max_ep_len'])
elif 'robot' in rl_params['platform']:
    from braille_rl.envs.robot.cont_ur5_braille_env.ur5GymEnv import UR5GymEnv
    env = UR5GymEnv(mode=rl_params['env_mode'], max_steps=rl_params['max_ep_len'])

sac(env, logger_kwargs=logger_kwargs,
         network_params=network_params,
         rl_params=rl_params,
         resume_training=False,
         resume_params=dict())
