from braille_rl.algos.disc_sac_algo.sac import *
from braille_rl.algos.rl_utils import load_model

# saved model to resume training from
model_dir = '../saved_models/sim/discrete/arrows/disc_sac/disc_sac_s1/'
# model_dir = '../saved_models/sim/discrete/alphabet/disc_sac/disc_sac_s1/'
model_save_name = 'tf1_save'

sess, model, logger_kwargs, rl_params, network_params, env, resume_state = load_model(model_dir, model_save_name)

# change some params for smooth transition (if resuming multiple times change this progress file)
logger_kwargs['output_fname'] = 'resumed_progress.txt'

resume_params = {'sess':sess,
                 'model':model,
                 'resume_state':resume_state,
                 'additional_epochs':10,
                }

print('')
print('Resuming Training')
print('')

# reload replay buffer and maybe current env state
sac(env, logger_kwargs=logger_kwargs,
         network_params=network_params,
         rl_params=rl_params,
         resume_training=True,
         resume_params=resume_params)
