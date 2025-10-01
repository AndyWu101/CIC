import gymnasium as gym
import torch
import numpy as np
from copy import deepcopy


from config import args
from model import Actor , Critic
from SAC import SAC
from replay_buffer import ReplayBuffer
from learning_curve import LearningCurve
from warm_up import warm_up
from evaluate import evaluate



###### 建立環境 ######
env = gym.make(args.env_name)


###### 設定隨機種子 ######
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset(seed=args.seed)
env.action_space.seed(args.seed)


###### 確定維度 ######
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high


###### 初始化 actor 和 critic ######
actor1 = Actor(state_dim, action_dim, max_action).to(args.device)
actor2 = deepcopy(actor1)
actor2_optimizer = torch.optim.Adam(actor2.parameters(), lr=args.actor_learning_rate)

critic = Critic(state_dim, action_dim).to(args.device)
critic_target = deepcopy(critic)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)


###### 初始化 SAC ######
sac = SAC(action_dim)


###### 初始化 replay buffer ######
replay_buffer = ReplayBuffer()


###### 初始化 learning curve ######
learning_curve = LearningCurve(actor1)


###### 初始化 Lambda buffer ######
Lambda = args.initial_Lambda
Lambda_buffer = []
for i in range(args.Lambda_buffer_size):
    Lambda_buffer.append((Lambda, -np.inf))


###### Warm Up ######
warm_up(env, replay_buffer, learning_curve)



###### 開始訓練 ######
while learning_curve.steps < args.max_steps:

    accumulated_evaluation_steps = 0

    evaluation_steps = evaluate(actor1, env, replay_buffer, learning_curve, 1)
    accumulated_evaluation_steps += evaluation_steps

    evaluation_steps = evaluate(actor2, env, replay_buffer, learning_curve, args.initial_evaluations)
    accumulated_evaluation_steps += evaluation_steps

    actor1_fitness = np.mean(actor1.fitness)
    actor2_fitness = np.mean(actor2.fitness)

    if actor1_fitness < actor2_fitness:
        actor1 = deepcopy(actor2)
        print("Changed")

    Lambda_buffer.append((Lambda, actor2_fitness))
    Lambda_buffer.pop(0)

    Lambda_buffer_ = deepcopy(Lambda_buffer)
    Lambda_buffer_.sort(key=lambda pair: pair[1], reverse=True)
    Lambda_mu = np.mean([pair[0] for pair in Lambda_buffer_[ : args.Lambda_buffer_size // 2]])

    Lambda = np.random.normal(loc=Lambda_mu, scale=args.Lambda_std)
    Lambda = np.clip(Lambda, 0, 1)

    for i in range(accumulated_evaluation_steps):

        sac.train(
            actor1,
            actor2,
            actor2_optimizer,
            critic,
            critic_target,
            critic_optimizer,
            replay_buffer,
            Lambda
        )

    actor2.fitness.clear()


    learning_curve.actor = actor1
    learning_curve.Lambda_mu = Lambda_mu

    if (learning_curve.steps % args.test_performance_freq == 0) and (learning_curve.steps <= args.max_steps):
        learning_curve.learning_curve_scores[-1] = learning_curve.test_performance(actor1)
        learning_curve.learning_curve_Lambda_mus[-1] = Lambda_mu


    print(f"{Lambda_mu:.3f}")
    print(f"a1 score: {actor1_fitness:.0f}   a2 score: {actor2_fitness:.0f}")
    print(f"steps={learning_curve.steps}  score={learning_curve.learning_curve_scores[-1]:.0f}")
    print("####################################")


###### 儲存結果 ######
if args.save_result == True:
    learning_curve.save()


print("Finish")


