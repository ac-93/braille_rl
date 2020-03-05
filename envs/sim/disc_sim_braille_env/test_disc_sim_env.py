import time
import random
import matplotlib.pyplot as plt

from envs.sim.disc_sim_braille_env.mockKBGymEnv import mockKBGymEnv

def main():
    seed = int(1)
    num_iter = 5
    env = mockKBGymEnv(mode='arrows', max_steps=50)

    env.seed(seed)
    env.action_space.np_random.seed(seed)
    random.seed(seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    for i in range(num_iter):

        goal = random.choice(env.goal_list)
        env.goal_button = goal

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        print('Goal Button: {}'.format(goal))
        time.sleep(2)

        while (not d):
            a = env.action_space.sample()
            o, r, d, info = env.step(a)
            ep_len+=1
            print('')
            print('act', env.MKB.action_list[a])
            print('rew', r)
            plt.imshow(o, cmap='gray')
            plt.show()


    env.close()

if __name__=="__main__":
    main()
