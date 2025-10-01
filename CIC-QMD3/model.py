import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_dim)

        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=args.device)

        self.fitness = []


    def forward(self, state):

        a = F.relu(self.linear1(state))
        a = F.relu(self.linear2(a))

        return self.max_action * torch.tanh(self.linear3(a))



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)


    def forward(self, state, action):

        s_a = torch.cat([state, action], dim=-1)

        q = F.relu(self.linear1(s_a))
        q = F.relu(self.linear2(q))
        q = self.linear3(q)

        return q


