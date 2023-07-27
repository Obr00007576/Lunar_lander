import gym
import copy
import torch
import random
from ReplayBuffer import ReplayBuffer
from DQNLL import DQNLL, device
import os

 
def main():
    epsilon = 0.001
    epsilon_decay = 0.99
    epsilon_min=0.001

    model_path = './LunarLander.pth'
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    observation, info = env.reset(seed=42) 
    model = DQNLL()
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    model_hat = copy.deepcopy(model)
    reward_sum=0
    replays = ReplayBuffer()
    n,c=0, 0

    while n < 10000:
        action = random.randint(0, 3) if random.random()<epsilon else model.get_action_value(observation)[0].item()
        new_observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        replays.add(observation, action, reward, new_observation)
        obs, acts, rews, nobs = replays.get_batch()
        model.train_batch(
            torch.FloatTensor(obs).to(device), 
            torch.FloatTensor(rews).to(device), 
            model_hat(torch.FloatTensor(nobs).to(device)), 
            torch.tensor(acts).to(device)
            )

        #print(f'y: {y}, pred: {torch.max(model(torch.FloatTensor(observation)))}')
        observation = new_observation
        if n%10==0:
            torch.save(model.state_dict(), model_path)

        if terminated or truncated:
            model_hat = copy.deepcopy(model)
            if reward_sum>100:
                epsilon*=epsilon_decay
                if epsilon < epsilon_min:
                    epsilon = epsilon_min
            n+=1
            print(f'epoch {n}: {reward_sum}')
            reward_sum=0
            observation, info = env.reset()
        c+=1
    env.close()

if __name__=='__main__':
    main()