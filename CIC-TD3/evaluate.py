import torch
import numpy as np

from config import args
from model import Actor
from replay_buffer import ReplayBuffer
from learning_curve import LearningCurve


def evaluate(actor: Actor, env, replay_buffer: ReplayBuffer, learning_curve: LearningCurve, test_n: int):

    evaluation_steps = 0

    max_action = env.action_space.high

    for t in range(test_n):

        if len(actor.fitness) < args.max_evaluations:

            score = 0

            state , _ = env.reset()
            done = False
            reach_step_limit = False

            while (not done) and (not reach_step_limit):

                with torch.no_grad():
                    state_ = torch.tensor(state, dtype=torch.float32, device=args.device)
                    action = actor(state_)
                    action = action.cpu().numpy()

                noise = np.random.normal(0, max_action * args.exploration_noise, size=action.shape)
                action = (action + noise).clip(-max_action , max_action)

                next_state , reward , done , reach_step_limit , _ = env.step(action)

                replay_buffer.push(state, action, next_state, reward, not done)

                score += reward

                evaluation_steps += 1
                learning_curve.add_step()

                state = next_state

            actor.fitness.append(score)

        else:

            break


    return evaluation_steps


