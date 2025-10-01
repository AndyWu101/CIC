import torch
import torch.nn.functional as F
import numpy as np


from config import args
from model import Actor , Critic
from replay_buffer import ReplayBuffer


class QMD3:

    def __init__(self, max_action: np.ndarray):

        self.train_steps = 0

        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=args.device)
        self.policy_noise = args.policy_noise * self.max_action
        self.policy_noise_clip = args.policy_noise_clip * self.max_action
        self.gamma = torch.tensor(args.gamma, dtype=torch.float32, device=args.device)
        self.tau = torch.tensor(args.tau, dtype=torch.float32, device=args.device)

        self.quasi_index = (args.critic_size // 2) - 1


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

        self.train_steps += 1


        replays = replay_buffer.sample()

        states = torch.stack([replay.state for replay in replays])
        actions = torch.stack([replay.action for replay in replays])
        rewards = torch.stack([replay.reward for replay in replays])
        next_states = torch.stack([replay.next_state for replay in replays])
        not_dones = torch.stack([replay.not_done for replay in replays])


        # 計算 target_Q
        with torch.no_grad():

            split_point = int(args.batch_size * Lambda)

            actor1_next_actions = actor1(next_states[ : split_point])
            actor2_next_actions = actor2(next_states[split_point : ])

            next_actions = torch.cat((actor1_next_actions, actor2_next_actions), dim=0)

            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.policy_noise_clip , self.policy_noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action , self.max_action)

            multi_next_Qs = torch.stack([critic_target(next_states, next_actions) for critic_target in critic_targets])
            multi_next_Qs, _ = multi_next_Qs.sort(dim=0)
            next_Qs = multi_next_Qs[self.quasi_index]

            target_Qs = rewards + not_dones * self.gamma * next_Qs


        for i in range(args.critic_size):

            Qs = critics[i](states, actions)

            critic_loss = F.mse_loss(Qs, target_Qs)

            critic_optimizers[i].zero_grad()
            critic_loss.backward()
            critic_optimizers[i].step()



        # 訓練 actor 並更新 target network
        if self.train_steps % args.policy_frequency == 0:

            actor_actions = actor2(states)

            multi_Qs = torch.stack([critic(states, actor_actions) for critic in critics])
            actor_loss = -multi_Qs.mean()

            actor2_optimizer.zero_grad()
            actor_loss.backward()
            actor2_optimizer.step()

            # 更新 target network
            with torch.no_grad():
                for critic, critic_target in zip(critics, critic_targets):
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


