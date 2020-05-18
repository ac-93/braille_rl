import os
import numpy as np
import time
import gym
import tensorflow as tf
from spinup.utils.logx import EpochLogger

from braille_rl.algos.plot_progress import plot_progress
from braille_rl.algos.rl_utils import *
from braille_rl.algos.dd_dqn_algo.core import *

# configure gpu use and supress tensorflow warnings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
Dueling Double Deep Q Network (DQN)
"""
def dd_dqn(env, logger_kwargs=dict(), network_params=dict(), rl_params=dict(), resume_training=False, resume_params=dict()):

    logger = EpochLogger(**logger_kwargs)

    if not resume_training:
        save_vars = locals().copy()
        save_vars.pop('env')
        logger.save_config(save_vars)

    # ==== control params ====
    seed                 = rl_params['seed']
    epochs               = rl_params['epochs']
    steps_per_epoch      = rl_params['steps_per_epoch']
    replay_size          = rl_params['replay_size']
    update_freq          = rl_params['update_freq']
    n_updates            = rl_params['n_updates']
    batch_size           = rl_params['batch_size']
    start_steps          = rl_params['start_steps']
    max_ep_len           = rl_params['max_ep_len']
    num_tests            = rl_params['num_tests']
    save_freq            = rl_params['save_freq']

    # ==== rl params ====
    use_HER              = rl_params['use_HER']
    use_prev_a           = rl_params['use_prev_a']
    gamma                = rl_params['gamma']
    polyak               = rl_params['polyak']
    q_lr                 = rl_params['q_lr']

    # ==== noise params ====
    act_noise_min        = rl_params['act_noise_min']
    act_noise_max        = rl_params['act_noise_max']
    act_noise_max_steps  = rl_params['act_noise_max_steps']
    test_act_noise       = rl_params['test_act_noise']

    # if resuming sess is passed as a param
    if not resume_training:
        sess = tf.compat.v1.Session(config=tf_config)

    # set seeding (still not perfectly deterministic)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    # get required gym spaces
    obs = env.observation_space
    act = env.action_space

    # get the size after resize
    obs_dim = network_params['input_dims']
    act_dim = act.n
    goal_dim = len(env.goal_list)

    # add action dimension to network params
    network_params['output_dim'] = act_dim

    if not resume_training:
        # init a state buffer for storing last m states
        train_state_buffer = StateBuffer(m=obs_dim[2])
        test_state_buffer  = StateBuffer(m=obs_dim[2])

        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim, size=replay_size)

        # Inputs to computation graph
        x_ph, a_ph, prev_a_ph, x2_ph, r_ph, d_ph, g_ph = placeholders(obs_dim, act_dim, act_dim, obs_dim, None, None, goal_dim)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            value_x1, advantage_x1, value_x2, advantage_x2 = action_value_networks(x_ph, x2_ph, use_prev_a, a_ph, prev_a_ph, g_ph, network_params)

        # Target networks
        with tf.variable_scope('target'):
            _, _, value_targ_x2, advantage_targ_x2 = action_value_networks(x_ph, x2_ph, use_prev_a, a_ph, prev_a_ph, g_ph, network_params)

        var_counts = tuple(count_vars(scope) for scope in ['main/q', 'target/q'])
        print("""\nNumber of parameters:
                   main q: %d
                   target q: %d \n"""%var_counts)

        # combine value and advantage functions
        q_x1 = value_x1 + tf.subtract(advantage_x1, tf.reduce_mean(advantage_x1, axis=1, keepdims=True))
        q_x2 = value_x2 + tf.subtract(advantage_x2, tf.reduce_mean(advantage_x2, axis=1, keepdims=True))
        q_targ = value_targ_x2 + tf.subtract(advantage_targ_x2, tf.reduce_mean(advantage_targ_x2, axis=1, keepdims=True))

        # get the index of the maximum q value, corresponds with action taken
        pi = tf.argmax(q_x1, axis=1)

        # get q values for actions taken
        q_val = tf.reduce_sum(tf.multiply(q_x1, a_ph), axis=1)

        # Double QL uses maximum action from main network but q value from target
        max_q_x2 = tf.one_hot(tf.argmax(q_x2, axis=1), depth=act_dim)
        q_targ_val = tf.reduce_sum(tf.multiply(q_targ, max_q_x2), axis=1)

        # Bellman backup for Q function
        q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_targ_val)

        # mean squared error loss
        q_loss = 0.5 * tf.reduce_mean((q_val - q_backup)**2)

        # set up optimizer
        trainable_vars = get_vars('main/q')
        q_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=q_lr, epsilon=1e-04)
        train_q_op = q_optimizer.minimize(q_loss, var_list=trainable_vars, name='train_q_op')

        # Polyak averaging for target variables (polyak=0.00 for hard update)
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))], name='target_update')

        # Initializing targets to match main variables
        target_init = tf.group([tf.compat.v1.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    else:
        # if resuming define all the ph and outputs from saved model
        # inputs
        x_ph      = resume_params['model']['x_ph']
        a_ph      = resume_params['model']['a_ph']
        prev_a_ph = resume_params['model']['prev_a_ph']
        x2_ph     = resume_params['model']['x2_ph']
        r_ph      = resume_params['model']['r_ph']
        d_ph      = resume_params['model']['d_ph']
        g_ph      = resume_params['model']['g_ph']

        # outputs
        pi     = resume_params['model']['pi']
        q_loss = resume_params['model']['q_loss']
        q_x1   = resume_params['model']['q_x1']

        # small buffers
        replay_buffer = resume_params['resume_state']['replay_buffer']
        train_state_buffer = resume_params['resume_state']['train_state_buffer']
        test_state_buffer = resume_params['resume_state']['test_state_buffer']

        # get needed operations from graph by name (trouble saving these)
        train_q_op    = tf.get_default_graph().get_operation_by_name("train_q_op")
        target_update = tf.get_default_graph().get_operation_by_name("target_update")

    if not resume_training:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(target_init)
    else:
        sess = resume_params['sess']

    # Setup model saving
    if save_freq is not None:
        logger.setup_tf_saver(sess, inputs={'x_ph':x_ph, 'a_ph':a_ph, 'prev_a_ph':prev_a_ph, 'x2_ph':x2_ph, 'r_ph':r_ph, 'd_ph':d_ph, 'g_ph':g_ph},
                                    outputs={'pi':pi, 'q_loss':q_loss, 'q_x1':q_x1})

    def get_action(state, one_hot_goal, prev_a, noise_scale):
        state = state.astype('float32') / 255.
        if np.random.random_sample() < noise_scale:
            a = env.action_space.sample()
        else:
            a = sess.run(pi, feed_dict={x_ph: [state],
                                        g_ph: [one_hot_goal],
                                        prev_a_ph: [prev_a]})[0]
        return a

    def reset(state_buffer):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        o = process_image_observation(o, obs_dim)
        r = process_reward(r)
        state = state_buffer.init_state(init_obs=o)
        prev_a = np.zeros(act_dim)

        # new random goal when the env is reset
        goal_id = np.random.randint(goal_dim)
        one_hot_goal = np.eye(goal_dim)[goal_id]
        goal = env.goal_list[goal_id]
        env.goal_button = goal
        # print('Goal Button: {}'.format(goal))

        return o, r, d, ep_ret, ep_len, state, one_hot_goal, prev_a

    def test_agent(n=1):
        print('Testing...')
        for j in range(n):
            test_o, test_r, test_d, test_ep_ret, test_ep_len, test_state, test_one_hot_goal, test_prev_a = reset(test_state_buffer)

            while not(test_d or (test_ep_len == max_ep_len)):

                test_a = get_action(test_state, test_one_hot_goal, test_prev_a, test_act_noise)

                test_o, test_r, test_d, _ = env.step(test_a)
                test_o = process_image_observation(test_o, obs_dim)
                test_r = process_reward(test_r)
                test_state = test_state_buffer.append_state(test_o)

                test_ep_ret += test_r
                test_ep_len += 1

                test_one_hot_a = process_action(test_a, act_dim)
                test_prev_a = test_one_hot_a

            logger.store(TestEpRet=test_ep_ret, TestEpLen=test_ep_len)

    # ================== Main training Loop  ==================
    if not resume_training:
        start_time = time.time()
        o, r, d, ep_ret, ep_len, state, one_hot_goal, prev_a = reset(train_state_buffer)

        total_steps = steps_per_epoch * epochs
        act_noise = update_eps(current_step=0, min_eps=act_noise_min, max_eps=act_noise_max, max_steps=act_noise_max_steps)
        resume_t = 0

        # array for storing states used with HER
        if use_HER:
            HER_buffer = HERBuffer(obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim, size=max_ep_len)

    # resuming training
    else:
        start_time  = time.time()
        total_steps = steps_per_epoch * (epochs +  resume_params['additional_epochs'])
        act_noise   = resume_params['resume_state']['act_noise']
        HER_buffer  = resume_params['resume_state']['HER_buffer']
        resume_t    = resume_params['resume_state']['resume_t']
        o, r, d, ep_ret, ep_len, state, one_hot_goal, prev_a = resume_params['resume_state']['rl_state']

        # reset the environment to the state set before saving
        env.set_env_state(resume_params['resume_state']['env_state'])

    # Main loop: collect experience in env and update/log each epoch
    for t in range(resume_t, total_steps):

        if t > start_steps:
            a = get_action(state, one_hot_goal, prev_a, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        o2        = process_image_observation(o2, obs_dim) # thresholding done in env
        r         = process_reward(r)
        one_hot_a = process_action(a, act_dim)

        next_state = train_state_buffer.append_state(o2)

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        # if life is lost then store done as true true
        replay_buffer.store(state, one_hot_a, prev_a, r, next_state, d, one_hot_goal)

        # append to HER buffer
        if use_HER:
            HER_buffer.store(state, one_hot_a, prev_a, r, next_state, d, one_hot_goal)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        state = next_state
        prev_a = one_hot_a

        # store additional states in replay buffer where the goal
        # is given by the final state, if the final state was incorrect
        if use_HER:
            if d and (ep_len != max_ep_len):

                # get actual goal achieved
                achieved_goal = np.eye(goal_dim)[env.goal_list.index(env.latest_button)]

                # if an incorrect goal was reached
                if (achieved_goal != one_hot_goal).any():

                    for j in range(ep_len):
                        # pull data from HER buffer
                        sample = HER_buffer.sample(j)

                        # change this to calc_rew function in env
                        if j == ep_len-1:
                            new_rew = env.max_rew
                        else:
                            new_rew = sample['rews']

                        # add to replay buffer
                        replay_buffer.store(sample['obs1'], sample['acts'], sample['prev_acts'], new_rew, sample['obs2'], sample['done'], achieved_goal)


        # do a single update
        if t > 0 and t % update_freq==0:
            for i in range(n_updates):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph:      batch['obs1'],
                             x2_ph:     batch['obs2'],
                             a_ph:      batch['acts'],
                             prev_a_ph: batch['prev_acts'],
                             r_ph:      batch['rews'],
                             d_ph:      batch['done'],
                             g_ph:      batch['goal']
                            }

                # Q-learning update
                outs = sess.run([q_loss, q_x1, train_q_op], feed_dict)
                logger.store(LossQ=outs[0], QVals=outs[1])

        if d or (ep_len == max_ep_len):
            # store episode values
            logger.store(EpRet=ep_ret, EpLen=ep_len)

            # reset the environment
            o, r, d, ep_ret, ep_len, state, one_hot_goal, prev_a = reset(train_state_buffer)

            if use_HER:
                # reset HER buffer
                HER_buffer.reset()

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:

            epoch = t // steps_per_epoch

            # update target network
            outs = sess.run(target_update)

            # update actor noise every epoch
            act_noise = update_eps(current_step=t, min_eps=act_noise_min, max_eps=act_noise_max, max_steps=act_noise_max_steps)

            # save everything neccessary for restarting training from current position
            env_state = env.get_env_state()

            # Save model
            if save_freq is not None:
                if (epoch % save_freq == 0) or (epoch == epochs-1):
                    print('Saving...')
                    rl_state = [o, r, d, ep_ret, ep_len, state, one_hot_goal, prev_a]
                    logger.save_state(state_dict={'env_state': env_state,
                                                  'replay_buffer':replay_buffer,
                                                  'train_state_buffer': train_state_buffer,
                                                  'test_state_buffer': test_state_buffer,
                                                  'HER_buffer': HER_buffer,
                                                  'act_noise': act_noise,
                                                  'resume_t': t+1,
                                                  'rl_state':rl_state
                                                  })
            # Test the performance of the deterministic version of the agent. (resets the env)
            test_agent(n=num_tests)

            # set params for resuming training
            env.set_env_state(env_state)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Actor Noise', act_noise)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    plot_progress(os.path.join(logger_kwargs['output_dir'],'progress.txt'), show_plot=False)

if __name__ == '__main__':
    pass
