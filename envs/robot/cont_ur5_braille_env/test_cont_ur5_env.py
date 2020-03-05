import random
from envs.robot.cont_ur5_braille_env.ur5GymEnv import UR5GymEnv

def main():
    seed = 1
    num_iter = 5
    env = UR5GymEnv(mode='arrows', max_steps=25)

    env.seed(seed)
    env.action_space.np_random.seed(seed)
    random.seed(seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_low  = env.action_space.low[0]
    act_high = env.action_space.high[0]

    for i in range(num_iter):

        goal = random.choice(env.goal_list)
        print('Goal Button: {}'.format(goal))
        env.goal_button = goal

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        while (not d):
            action = env.action_space.sample()
            o, r, d, info = env.step(action)
            ep_len+=1

    env.close()

if __name__=="__main__":
    main()
