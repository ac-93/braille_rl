import time
import numpy as np
import random
import matplotlib.pyplot as plt

from braille_rl.envs.sim.cont_sim_braille_env.mockKBGymEnv import mockKBGymEnv

def main():
    seed = int(1)
    num_iter = 5
    env = mockKBGymEnv(mode='arrows', max_steps=25)

    env.seed(seed)
    env.action_space.np_random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_low  = env.action_space.low[0]
    act_high = env.action_space.high[0]

    for i in range(num_iter):

        goal = random.choice(env.goal_list)
        env.goal_button = goal

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        print('Goal Button: {}'.format(goal))
        time.sleep(1)

        plt.imshow(o, cmap='gray')
        plt.show()

        while (not d):
            a = env.action_space.sample()
            o, r, d, info = env.step(a)
            ep_len+=1
            print('rew', r)
            print('act', a)
            plt.imshow(o, cmap='gray')
            plt.show()


    env.close()

if __name__=="__main__":
    main()
