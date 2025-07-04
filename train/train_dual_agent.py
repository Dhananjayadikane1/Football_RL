import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from env.football_env import FootballEnv
from agents.ppo_goalkeeper import PPOGoalkeeperAgent
from agents.dqn_striker import DQNStrikerAgent
from visualize.plot_learning_curve import plot_learning_curve
from visualize.game_visualizer import visualize_penalty


def main():
    env = FootballEnv()

    state_dim = 5  # includes shot_angle
    action_dim_keeper = 3    # [left, stay, right]
    action_dim_striker = 10  # 10 shot buckets across goal

    dqn_striker = DQNStrikerAgent(state_dim, action_dim_striker)
    ppo_keeper = PPOGoalkeeperAgent(state_dim, action_dim_keeper)

    num_episodes = 500
    reward_striker_history = []
    reward_keeper_history = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()

        # Striker chooses target
        striker_action_idx = dqn_striker.choose_action(state)
        striker_target_x = np.linspace(0.5, 6.8, action_dim_striker)[striker_action_idx]

        # Keeper chooses action
        keeper_action_idx, keeper_prob = ppo_keeper.choose_action(state)

        # Take a step
        next_state, reward_striker, reward_keeper, result, done = env.step(striker_target_x, keeper_action_idx)

        # Store transitions
        dqn_striker.store_transition(state, striker_action_idx, reward_striker, next_state, done)
        ppo_keeper.store_transition((state, keeper_action_idx, reward_keeper, next_state, float(done), keeper_prob))

        # Learning
        if episode % 10 == 0:
            dqn_striker.learn()
        if episode % 50 == 0:
            ppo_keeper.learn()

        # Log rewards
        reward_striker_history.append(reward_striker)
        reward_keeper_history.append(reward_keeper)

        # Output progress
        if episode % 50 == 0:
            print(f"Ep {episode}/{num_episodes} | S: {reward_striker:.2f} | K: {reward_keeper:.2f} | Result: {result}")
            visualize_penalty(striker_target_x, state[0], result)

    # Plot learning curve
    plot_learning_curve(reward_striker_history, reward_keeper_history)


if __name__ == '__main__':
    main()
