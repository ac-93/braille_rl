import os
import numpy as np
import tensorflow as tf
import random
import csv
from spinup.utils.logx import restore_tf_graph
from sklearn.metrics import confusion_matrix

from braille_rl.envs.sim.cont_sim_braille_env.mockKBGymEnv import mockKBGymEnv
from braille_rl.envs.robot.cont_ur5_braille_env.ur5GymEnv import UR5GymEnv
from braille_rl.algos.rl_utils import *
from braille_rl.algos.image_utils import load_json_obj, permutation, plot_confusion_matrix

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# saved model to evaluate
model_dir = '../saved_models/sim/continuous/arrows/cont_sac/cont_sac_s1/'
# model_dir = '../saved_models/sim/continuous/alphabet/cont_sac/cont_sac_s1/'
model_save_name = 'tf1_save'

# set up the trained model
sess = tf.compat.v1.Session(config=tf_config)
model  = restore_tf_graph(sess=sess, fpath=os.path.join(model_dir, model_save_name))
config = load_json_obj(os.path.join(model_dir, 'config'))

if 'sim' in config['rl_params']['platform']:
    env = mockKBGymEnv(mode=config['rl_params']['env_mode'], max_steps=config['rl_params']['max_ep_len'])
elif 'robot' in config['rl_params']['platform']:
    env = UR5GymEnv(mode=config['rl_params']['env_mode'], max_steps=config['rl_params']['max_ep_len'])

# define neccesry inputs/outputs
x_ph = model['x_ph']
g_ph = model['g_ph']
prev_a_ph = model['prev_a_ph']
pi = model['pi']
mu = model['mu']

obs_dim = config['network_params']['input_dims']
test_state_buffer = StateBuffer(m=obs_dim[2])
max_ep_len = config['rl_params']['max_ep_len']

act_dim = env.action_space.shape[0]
goal_dim = len(env.goal_list)

# set seeding
seed=1
tf.set_random_seed(seed)
np.random.seed(seed)
env.seed(seed)
env.action_space.np_random.seed(seed)
random.seed(seed)

# create list of key sequences to be typed
if 'arrows' in config['rl_params']['env_mode']:
    test_goal_list = permutation(['UP', 'DOWN', 'LEFT', 'RIGHT'])

elif 'alphabet' in config['rl_params']['env_mode']:
    test_goal_list = [random.sample(env.goal_list,  len(env.goal_list)) for i in range(10)]

print('Key Sequences: ')
for sequence in test_goal_list:
    print(sequence)
print('')

def get_action(state, one_hot_goal, prev_a, deterministic=False):
    state = state.astype('float32') / 255.
    act_op = mu if deterministic else pi
    return sess.run(act_op, feed_dict={x_ph: [state],
                                       g_ph: [one_hot_goal],
                                       prev_a_ph: [prev_a]})[0]

def reset(state_buffer, goal):
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    o = process_image_observation(o, obs_dim)
    r = process_reward(r)
    state = state_buffer.init_state(init_obs=o)
    prev_a = np.zeros(act_dim)


    # new random goal when the env is reset
    goal_id = env.goal_list.index(goal)
    one_hot_goal = np.eye(goal_dim)[goal_id]
    env.goal_button = goal

    return o, r, d, ep_ret, ep_len, state, one_hot_goal, prev_a


def test_agent(sequence):
    n=len(sequence)

    correct_count = 0
    step_count = 0

    goal_list = []
    achieved_goal_list = []

    for j in range(n):
        test_o, test_r, test_d, test_ep_ret, test_ep_len, test_state, test_one_hot_goal, test_prev_a = reset(test_state_buffer, sequence[j])

        while not(test_d or (test_ep_len == max_ep_len)):

            test_a = get_action(test_state, test_one_hot_goal, test_prev_a, False)

            test_o, test_r, test_d, _ = env.step(test_a)
            test_o = process_image_observation(test_o, obs_dim)
            test_r = process_reward(test_r)
            test_state = test_state_buffer.append_state(test_o)

            test_ep_ret += test_r
            test_ep_len += 1

            test_prev_a = test_a

            if test_r == 1:
                correct_count += 1

            if test_d:
                achieved_goal_list.append(env.latest_button)
                goal_list.append(sequence[j])


        step_count += test_ep_len

    acc = correct_count/n

    print('Sequence Steps: {}'.format(step_count))
    print('Sequence Accuracy: {}'.format(acc))

    return correct_count, n, step_count, goal_list, achieved_goal_list


csv_dir = os.path.join(model_dir, 'evaluation_output.csv')

with open(csv_dir, "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Episode','Example Sequence', 'Typed Sequence'])

    achieved_goals = []
    goals = []
    total_correct = 0
    total_count = 0
    total_steps = 0
    for (i,sequence) in enumerate(test_goal_list):
        correct, num_elements, step_count, goal_list, achieved_goal_list = test_agent(sequence)

        total_correct += correct
        total_count += num_elements
        total_steps += step_count

        writer.writerow([i, goal_list, achieved_goal_list])

        achieved_goals.append(achieved_goal_list)
        goals.append(goal_list)

    overall_acc = total_correct/total_count
    print('')
    print('Total Steps:   ', total_steps)
    print('Total Count:   ', total_count)
    print('Total Correct: ', total_correct)
    print('Overall Acc:   ', overall_acc)

    writer.writerow([])
    writer.writerow(['Total Steps', 'Total Count', 'Total Correct', 'Overall Acc'])
    writer.writerow([total_steps, total_count, total_correct, overall_acc])

goals = np.hstack(goals)
achieved_goals = np.hstack(achieved_goals)
cnf_matrix = confusion_matrix(goals, achieved_goals)
plot_confusion_matrix(cnf_matrix, classes=env.goal_list, normalize=True, cmap=plt.cm.Blues,
                      title='Normalized Confusion matrix', dirname=None, save_flag=False)

env.close()
