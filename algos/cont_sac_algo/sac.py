import os
import numpy as np
import time
import gym
import tensorflow as tf

from algos.plot_progress import plot_progress
from algos.rl_utils import *
from algos.cont_sac_algo.core import *
from spinup.utils.logx import EpochLogger

# configure gpu use and supress tensorflow warnings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
Soft Actor Critic
"""
def sac(env, logger_kwargs=dict(), network_params=dict(), rl_params=dict(), resume_training=False, resume_params=dict()):

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
    act_lr               = rl_params['act_lr']
    crit_lr              = rl_params['crit_lr']
    alph_lr              = rl_params['alph_lr']

    # ==== exploration params ====
    alpha                = rl_params['alpha']
    target_entropy       = rl_params['target_entropy']

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

        # alpha Params
        if target_entropy == 'auto':
            target_entropy = tf.cast(-act_dim, tf.float32)
        else:
            target_entropy = tf.cast(target_entropy, tf.float32)

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)

        if alpha == 'auto': # auto tune alpha
            alpha = tf.exp(log_alpha)
        else: # fixed alpha
            alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=alpha)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            mu, pi, logp_pi, q1_a, q2_a  = create_rl_networks(x_ph, a_ph, use_prev_a, prev_a_ph, g_ph, act_high, network_params)

        with tf.variable_scope('main', reuse=True):
            # compose q with pi, for pi-learning
            _, _, _, q1_pi, q2_pi = create_rl_networks(x_ph, pi, use_prev_a, prev_a_ph, g_ph, act_high, network_params)

            # get actions and log probs of actions for next states, for Q-learning
            _, pi_next, logp_pi_next, _, _ = create_rl_networks(x2_ph, a_ph, use_prev_a, prev_a_ph, g_ph, act_high, network_params)

        # Target networks
        with tf.variable_scope('target'):
            _, _, _, q1_pi_targ, q2_pi_targ = create_rl_networks(x2_ph, pi_next, use_prev_a, a_ph, g_ph, act_high, network_params)

        var_counts = tuple(count_vars(scope) for scope in ['log_alpha',
                                                           'main/pi',
                                                           'main/q1',
                                                           'main/q2',
                                                           'main'])
        print("""\nNumber of parameters:
                 alpha: %d,
                 pi: %d,
                 q1: %d,
                 q2: %d,
                 total: %d\n"""%var_counts)

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)
        min_q_pi_targ = tf.minimum(q1_pi_targ, q2_pi_targ)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*(min_q_pi_targ - alpha*logp_pi_next))

        # critic losses
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_a)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_a)**2)
        value_loss = q1_loss + q2_loss

        # Soft actor losses
        pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)

        # alpha loss for temperature parameter
        alpha_backup = tf.stop_gradient(logp_pi + target_entropy)
        alpha_loss = -tf.reduce_mean((log_alpha * alpha_backup))

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=act_lr, epsilon=1e-04)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'), name='train_pi_op')

        # Value train op
        value_optimizer = tf.train.AdamOptimizer(learning_rate=crit_lr, epsilon=1e-04)
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss, var_list=get_vars('main/q'), name='train_value_op')

        # Alpha train op
        alpha_optimizer = tf.train.AdamOptimizer(learning_rate=alph_lr, epsilon=1e-04)
        with tf.control_dependencies([train_value_op]):
            train_alpha_op = alpha_optimizer.minimize(alpha_loss, var_list=get_vars('log_alpha'), name='train_alpha_op')

        # Polyak averaging for target variables
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))], name='target_update')

        # Initializing targets to match main variables
        target_init = tf.group([tf.compat.v1.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        sess.run(tf.compat.v1.global_variables_initializer())
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
        mu             = resume_params['model']['mu']
        pi             = resume_params['model']['pi']
        pi_loss        = resume_params['model']['pi_loss']
        q1_loss        = resume_params['model']['q1_loss']
        q2_loss        = resume_params['model']['q2_loss']
        q1_a           = resume_params['model']['q1_a']
        q2_a           = resume_params['model']['q2_a']
        logp_pi        = resume_params['model']['logp_pi']
        target_entropy = resume_params['model']['target_entropy']
        alpha_loss     = resume_params['model']['alpha_loss']
        alpha          = resume_params['model']['alpha']

        # buffers
        replay_buffer = resume_params['resume_state']['replay_buffer']
        train_state_buffer = resume_params['resume_state']['train_state_buffer']
        test_state_buffer = resume_params['resume_state']['test_state_buffer']

        # get needed operations from graph by name (trouble saving these)
        train_pi_op    = tf.get_default_graph().get_operation_by_name("train_pi_op")
        train_value_op = tf.get_default_graph().get_operation_by_name("train_value_op")
        train_alpha_op = tf.get_default_graph().get_operation_by_name("train_alpha_op")
        target_update  = tf.get_default_graph().get_operation_by_name("target_update")

        sess = resume_params['sess']

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, q1_a, q2_a,
                logp_pi, target_entropy,
                alpha_loss, alpha,
                train_pi_op, train_value_op, train_alpha_op, target_update]

    # Setup model saving
    if save_freq is not None:
        logger.setup_tf_saver(sess, inputs={'x_ph': x_ph, 'a_ph': a_ph, 'prev_a_ph': prev_a_ph, 'x2_ph': x2_ph, 'r_ph': r_ph, 'd_ph': d_ph, 'g_ph': g_ph},
                                    outputs={'mu': mu, 'pi': pi, 'pi_loss': pi_loss, 'q1_loss': q1_loss, 'q2_loss': q2_loss,
                                             'q1_a': q1_a, 'q2_a': q2_a, 'logp_pi': logp_pi, 'target_entropy': target_entropy, 'alpha_loss': alpha_loss, 'alpha': alpha})

    def get_action(state, one_hot_goal, prev_a, deterministic=False):
        state = state.astype('float32') / 255.
        act_op = mu if deterministic else pi
        a = sess.run(act_op, feed_dict={x_ph: [state],
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

                test_a = get_action(test_state, test_one_hot_goal, test_prev_a, True)

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
        resume_t = 0

        # array for storing states used with HER
        if use_HER:
            HER_buffer = ContHERBuffer(obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim, size=max_ep_len)

    # resuming training
    else:
        start_time  = time.time()
        total_steps = steps_per_epoch * (epochs +  resume_params['additional_epochs'])
        HER_buffer  = resume_params['resume_state']['HER_buffer']
        resume_t    = resume_params['resume_state']['resume_t']
        o, r, d, ep_ret, ep_len, state, one_hot_goal, prev_a = resume_params['resume_state']['rl_state']

        # reset the environment to the state set before saving
        env.set_env_state(resume_params['resume_state']['env_state'])

    # Main loop: collect experience in env and update/log each epoch
    for t in range(resume_t, total_steps):

        if t > start_steps:
            a = get_action(state, one_hot_goal, prev_a, False)
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
        # if life is lost then store done as true true
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

                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0],
                             LossQ1=outs[1],    LossQ2=outs[2],
                             Q1Vals=outs[3],    Q2Vals=outs[4],
                             LogPPi=outs[5],    TargEntropy=outs[6],
                             LossAlpha=outs[7], Alpha=outs[8])

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
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPPi', average_only=True)
            logger.log_tabular('TargEntropy', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    plot_progress(os.path.join(logger_kwargs['output_dir'],'progress.txt'), show_plot=False)

if __name__ == '__main__':
    pass
