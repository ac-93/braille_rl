import os
import numpy as np
import time
import gym
import tensorflow as tf
from spinup.utils.logx import EpochLogger

from braille_rl.algos.plot_progress import plot_progress
from braille_rl.algos.rl_utils import *
from braille_rl.algos.td3_algo.core import *

# configure gpu use and supress tensorflow warnings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
"""
def td3(env, logger_kwargs=dict(), network_params=dict(), rl_params=dict(), resume_training=False, resume_params=dict()):

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
    act_lr               = rl_params['act_lr']
    crit_lr              = rl_params['crit_lr']
    update_after         = rl_params['update_after']
    update_every         = rl_params['update_every']
    policy_delay         = rl_params['policy_delay']

    # ==== noise params ====
    act_noise_min        = rl_params['act_noise_min']
    act_noise_max        = rl_params['act_noise_max']
    act_noise_max_steps  = rl_params['act_noise_max_steps']
    target_noise         = rl_params['target_noise']
    noise_clip           = rl_params['noise_clip']
    test_act_noise       = rl_params['test_act_noise']

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

    # get the obs size after resize of raw image
    obs_dim = network_params['input_dims']
    act_dim = env.action_space.shape[0]
    act_low  = env.action_space.low[0]
    act_high = env.action_space.high[0]
    goal_dim = len(env.goal_list)

    if not resume_training:

        # init a state buffer for storing last m states
        train_state_buffer = StateBuffer(m=obs_dim[2])
        test_state_buffer  = StateBuffer(m=obs_dim[2])

        # Experience buffer
        replay_buffer = ContReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim, size=replay_size)

        # Inputs to computation graph
        x_ph, a_ph, prev_a_ph, x2_ph, r_ph, d_ph, g_ph = placeholders(obs_dim, act_dim, act_dim, obs_dim, None, None, goal_dim)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            pi, q1, q2, q1_pi = create_rl_networks(x_ph, a_ph, use_prev_a, prev_a_ph, g_ph, act_high, network_params)

        # Target networks
        with tf.variable_scope('target'):
            # Note that the action placeholder going to actor_critic here is
            # irrelevant, because we only need q_targ(s, pi_targ(s)).
            # but prev_a_ph becomes a_ph
            pi_targ, _, _, _  = create_rl_networks(x2_ph, a_ph, use_prev_a, a_ph, g_ph, act_high, network_params)

        with tf.variable_scope('target', reuse=True):
            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, act_low, act_high)

            _, q1_targ, q2_targ, _  = create_rl_networks(x2_ph, a2, use_prev_a, a_ph, g_ph, act_high, network_params)

        var_counts = tuple(count_vars(scope) for scope in ['main/pi',
                                                           'main/q1',
                                                           'main/q2',
                                                           'main'])
        print("""\nNumber of parameters:
                 pi: %d,
                 q1: %d,
                 q2: %d,
                 total: %d\n"""%var_counts)

        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

        # TD3 losses
        pi_loss = -tf.reduce_mean(q1_pi)
        q1_loss = tf.reduce_mean((q1-backup)**2)
        q2_loss = tf.reduce_mean((q2-backup)**2)
        q_loss = q1_loss + q2_loss

        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=act_lr, epsilon=1e-04)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=crit_lr, epsilon=1e-04)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'), name='train_pi_op')
        train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'), name='train_q_op')

        # Polyak averaging for target variables
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))], name='target_update')

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        sess.run(tf.global_variables_initializer())
        sess.run(target_init)

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
        pi      = resume_params['model']['pi']
        q1      = resume_params['model']['q1']
        q2      = resume_params['model']['q2']
        q1_pi   = resume_params['model']['q1_pi']
        pi_targ = resume_params['model']['pi_targ']
        q1_targ = resume_params['model']['q1_targ']
        q2_targ = resume_params['model']['q2_targ']
        pi_loss = resume_params['model']['pi_loss']
        q_loss  = resume_params['model']['q_loss']

        # buffers
        replay_buffer = resume_params['resume_state']['replay_buffer']
        train_state_buffer = resume_params['resume_state']['train_state_buffer']
        test_state_buffer = resume_params['resume_state']['test_state_buffer']

        # get needed operations from graph by name (trouble saving these)
        train_pi_op   = tf.get_default_graph().get_operation_by_name("train_pi_op")
        train_q_op    = tf.get_default_graph().get_operation_by_name("train_q_op")
        target_update = tf.get_default_graph().get_operation_by_name("target_update")

        sess = resume_params['sess']

    # Setup model saving
    if save_freq is not None:
        logger.setup_tf_saver(sess, inputs={'x_ph': x_ph, 'a_ph': a_ph, 'prev_a_ph': prev_a_ph, 'x2_ph': x2_ph, 'r_ph': r_ph, 'd_ph': d_ph, 'g_ph': g_ph},
                                    outputs={'pi': pi, 'q1': q1, 'q2': q2, 'q1_pi': q1_pi, 'pi_targ': pi_targ, 'q1_targ': q1_targ, 'q2_targ': q2_targ, 'pi_loss': pi_loss, 'q_loss': q_loss})

    def get_action(state, one_hot_goal, prev_a, noise_scale=0.1):
        state = state.astype('float32') / 255.
        a = sess.run(pi, feed_dict={x_ph: [state],
                                    g_ph: [one_hot_goal],
                                    prev_a_ph: [prev_a]})[0]

        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, act_low, act_high)

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

        return o, r, d, ep_ret, ep_len, state, one_hot_goal, prev_a

    def test_agent(n=1):
        print('Testing...')
        for j in range(n):
            test_o, test_r, test_d, test_ep_ret, test_ep_len, test_state, test_one_hot_goal, test_prev_a = reset(test_state_buffer)

            while not(test_d or (test_ep_len == max_ep_len)):

                test_a = get_action(test_state, test_one_hot_goal, test_prev_a, noise_scale=0)

                test_o, test_r, test_d, _ = env.step(test_a)
                test_o = process_image_observation(test_o, obs_dim)
                test_r = process_reward(test_r)
                test_state = test_state_buffer.append_state(test_o)

                test_ep_ret += test_r
                test_ep_len += 1

                test_prev_a = test_a

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
            HER_buffer = ContHERBuffer(obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim, size=max_ep_len)

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
            a = get_action(state, one_hot_goal, prev_a, noise_scale=act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        o2 = process_image_observation(o2, obs_dim) # thresholding done in env
        r  = process_reward(r)

        next_state = train_state_buffer.append_state(o2)

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(state, a, prev_a, r, next_state, d, one_hot_goal)

        # append to HER buffer
        if use_HER:
            HER_buffer.store(state, a, prev_a, r, next_state, d, one_hot_goal)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        state = next_state
        prev_a = a

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

                        # change the reward to that of correct reward
                        if j == ep_len-1:
                            new_rew = env.max_rew
                        else:
                            new_rew = sample['rews']

                        # add to replay buffer
                        replay_buffer.store(sample['obs1'], sample['acts'], sample['prev_acts'], new_rew, sample['obs2'], sample['done'], achieved_goal)

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph:      batch['obs1'],
                             x2_ph:     batch['obs2'],
                             a_ph:      batch['acts'],
                             prev_a_ph: batch['prev_acts'],
                             r_ph:      batch['rews'],
                             d_ph:      batch['done'],
                             g_ph:      batch['goal']
                            }
                q_step_ops = [q_loss, q1, q2, train_q_op]
                outs = sess.run(q_step_ops, feed_dict)
                logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                if j % policy_delay == 0:
                    # Delayed policy update
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])


        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)

            # reset the environment
            o, r, d, ep_ret, ep_len, state, one_hot_goal, prev_a = reset(train_state_buffer)

            if use_HER:
                # reset HER buffer
                HER_buffer.reset()

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

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

            # Test the performance of the deterministic version of the agent.
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
            logger.log_tabular('Q1Vals', average_only=True)
            logger.log_tabular('Q2Vals', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Actor Noise', act_noise)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    plot_progress(os.path.join(logger_kwargs['output_dir'],'progress.txt'), show_plot=False)

if __name__ == '__main__':
    pass
