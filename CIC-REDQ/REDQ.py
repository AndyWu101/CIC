import torch
import torch.nn.functional as F
import numpy as np


from config import args
from model import Actor , Critic
from replay_buffer import ReplayBuffer


class REDQ:

    def __init__(self, action_dim):

        self.alpha = torch.tensor(args.alpha, dtype=torch.float32, device=args.device)
        self.gamma = torch.tensor(args.gamma, dtype=torch.float32, device=args.device)
        self.tau = torch.tensor(args.tau, dtype=torch.float32, device=args.device)

        if args.adaptive_alpha == True:

            mbpo_target_entropy_dict = {"Hopper": -1, "HalfCheetah": -3, "Walker2d": -3, "Ant": -4, "Humanoid": -2}

            if (args.mbpo_target_entropy == True) and (args.env_name[ : -3] in mbpo_target_entropy_dict):
                target_entropy = mbpo_target_entropy_dict[args.env_name[ : -3]]
            else:
                target_entropy = -action_dim

            self.target_entropy = torch.tensor(target_entropy, dtype=torch.float32, device=args.device)
            self.log_alpha = torch.tensor(0, requires_grad=True, dtype=torch.float32, device=args.device)
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.critic_learning_rate)
            self.alpha = self.log_alpha.detach().exp()


    def train(
            self,
            actor1: Actor,
            actor2: Actor,
            actor2_optimizer: torch.optim.Adam,
            critics: list[Critic],
            critic_targets: list[Critic],
            critic_optimizers: list[torch.optim.Adam],
            replay_buffer: ReplayBuffer,
            Lambda: float
    ):

        replays = replay_buffer.sample()

        states = torch.stack([replay.state for replay in replays])
        actions = torch.stack([replay.action for replay in replays])
        rewards = torch.stack([replay.reward for replay in replays])
        next_states = torch.stack([replay.next_state for replay in replays])
        not_dones = torch.stack([replay.not_done for replay in replays])

        # 計算 target_Q
        with torch.no_grad():

            split_point = int(args.batch_size * Lambda)

            actor1_next_actions, actor1_next_log_prob_pis = actor1.sample(next_states[ : split_point])
            actor2_next_actions, actor2_next_log_prob_pis = actor2.sample(next_states[split_point : ])

            next_actions = torch.cat((actor1_next_actions, actor2_next_actions), dim=0)
            next_log_prob_pis = torch.cat((actor1_next_log_prob_pis, actor2_next_log_prob_pis), dim=0)

            subset_indices = np.random.choice(args.critic_size, args.subset_size, replace=False)
            multi_next_Qs = []
            for index in subset_indices:
                next_Qs = critic_targets[index](next_states, next_actions)  # shape = (batch_size , 1)
                multi_next_Qs.append(next_Qs)

            multi_next_Qs = torch.cat(multi_next_Qs, dim=1)           # shape = (batch_size , subset_size)
            min_next_Qs , _ = multi_next_Qs.min(dim=1, keepdim=True)  # shape = (batch_size , 1)

            target_Qs = rewards + not_dones * self.gamma * (min_next_Qs - self.alpha * next_log_prob_pis)


        for i in range(args.critic_size):

            Qs = critics[i](states, actions)

            critic_loss = F.mse_loss(Qs, target_Qs)

            critic_optimizers[i].zero_grad()
            critic_loss.backward()
            critic_optimizers[i].step()

            # 更新 target network
            with torch.no_grad():
                for param, target_param in zip(critics[i].parameters(), critic_targets[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        # 訓練 actor
        actor_actions, log_prob_pis = actor2.sample(states)

        multi_Qs = []
        for critic in critics:
            Qs = critic(states, actor_actions)  # shape = (batch_size , 1)
            multi_Qs.append(Qs)

        multi_Qs = torch.cat(multi_Qs, dim=1)         # shape = (batch_size , critic_size)
        mean_Qs = multi_Qs.mean(dim=1, keepdim=True)  # shape = (batch_size , 1)

        actor_loss = (self.alpha * log_prob_pis - mean_Qs).mean()

        actor2_optimizer.zero_grad()
        actor_loss.backward()
        actor2_optimizer.step()


        # 更新 log_alpha
        if args.adaptive_alpha == True:

            log_alpha_loss = -self.log_alpha * (log_prob_pis.detach().mean() + self.target_entropy)

            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss.backward()
            self.log_alpha_optimizer.step()

            self.alpha = self.log_alpha.detach().exp()


