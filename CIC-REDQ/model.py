import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from config import args


LOG_STD_MAX = torch.tensor(2, dtype=torch.float32, device=args.device)
LOG_STD_MIN = torch.tensor(-20, dtype=torch.float32, device=args.device)
ln2 = torch.tensor(2, dtype=torch.float32, device=args.device).log()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mu_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=args.device)

        self.fitness = []


    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        return self.max_action * torch.tanh(self.mu_linear(x))


    def sample(self, state):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mu = self.mu_linear(x)
        log_std = self.log_std_linear(x).clamp(LOG_STD_MIN , LOG_STD_MAX)
        std = log_std.exp()

        normal = Normal(mu, std)
        a = normal.rsample()
        action = self.max_action * a.tanh()

        log_prob = normal.log_prob(a).sum(dim=-1, keepdim=True)
        log_prob_pi = log_prob - 2 * (ln2 - a - F.softplus(-2 * a)).sum(dim=-1, keepdim=True)  # 等價log(1 - tanh(a) ** 2)

        return action, log_prob_pi



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


