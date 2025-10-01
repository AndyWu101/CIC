from config import args
from replay_buffer import ReplayBuffer
from learning_curve import LearningCurve


def warm_up(env, replay_buffer: ReplayBuffer, learning_curve: LearningCurve, start_steps: int=args.start_steps):

    state , _ = env.reset()

    while learning_curve.steps < start_steps:

        action = env.action_space.sample()

        next_state , reward , done , reach_step_limit , _ = env.step(action)

        replay_buffer.push(state, action, next_state, reward, not done)

        learning_curve.add_step()

        if done or reach_step_limit:
            state , _ = env.reset()
        else:
            state = next_state


