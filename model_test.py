import gym
import torch
from DQNLL import DQNLL
import os
import random

def main():
    epsilon = 0
    model_path = './LunarLander.pth'
    env = gym.make("LunarLander-v2", render_mode='human')

    observation, info = env.reset() 
    model = DQNLL()
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("Can't find the model.")
        exit()
    reward_sum=0
    n=0

    while n <50:
        action = random.randint(0, 3) if random.random()<epsilon else model.get_action_value(observation)[0].item()
        #action = model.get_action_value(observation)[0].item()
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward

        if terminated or truncated:
            n+=1
            print(f'epoch {n}: {reward_sum}')
            reward_sum=0
            observation, info = env.reset()
    env.close()

if __name__=='__main__':
    main()