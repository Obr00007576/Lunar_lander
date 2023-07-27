import torch
from torch import nn
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {torch.cuda.get_device_name(0)}.")

action_space_len = 4
status_space_len = 8
discount_factor = 0.99

class DQNLL(nn.Module):
    def __init__(self) -> None:
        super(DQNLL, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=status_space_len, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=action_space_len)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def train_batch(self, obs, rews, predict_hat, acts):
        pred = self(obs)
        targets = self(obs)
        y = discount_factor*torch.amax(predict_hat, dim=1) + rews
        #targets[[i for i in range(len(acts))], acts] = y
        targets[[i for i in range(len(acts))], acts] = y
        # targets[[i for i in range(len(acts))], acts] = y
        # for i in range(len(targets)):
        #     targets[i, acts[i]] = y[i, 0]
        targets.to(device)
        loss = self.loss_fn(pred, targets)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def get_action_value(self, observation):
        pred = self(torch.FloatTensor(observation).to(device))
        action = torch.argmax(pred)
        pred_reward = pred[action]
        return action,pred_reward

# model = DQNLL()
# vec = torch.ones(2, 8)
# vec[1, 2]=34
# print(model(vec))
# a = model(vec)
# for i in range(100):
#     model.train_batch(vec, torch.FloatTensor([[2], [4]]))
# print(model(vec))