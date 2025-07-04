import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from env.football_env import FootballEnv
from agents.ppo_goalkeeper import PPOGoalkeeperAgent
import matplotlib.pyplot as plt


def main():
    env = FootballEnv()
    state_dim = 5  # [keeper_x, ball_x, ball_y, goal_center, shot_angle]
    action_dim = 3  # [left, stay, right]
    ppo_keeper = PPOGoalkeeperAgent(state_dim, action_dim)

    fixed_shot_positions = [1.5, 3.66, 5.8]  # left, center, right
    num_episodes = 500
    reward_history = []
    avg_reward_history = []

    for episode in range(1, num_episodes + 1):
        striker_target = np.random.choice(fixed_shot_positions)
        state = env.reset()

        action_idx, prob = ppo_keeper.choose_action(state)
        
        # Ensure env.step() returns exactly 5 items
        next_state, _, reward_keeper, result, done = env.step(striker_target, action_idx)

        ppo_keeper.store_transition((state, action_idx, reward_keeper, next_state, float(done), prob))

        reward_history.append(reward_keeper)

        if episode % 50 == 0:
            ppo_keeper.learn()
            avg_reward = np.mean(reward_history[-50:])
            avg_reward_history.append(avg_reward)
            print(f"Episode {episode}/{num_episodes} | Avg Keeper Reward (last 50): {avg_reward:.2f} | Result: {result}")

    # Plotting
    episodes = np.arange(50, num_episodes + 1, 50)
    plt.plot(episodes, avg_reward_history, marker='o')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (per 50 episodes)')
    plt.title('PPO Goalkeeper Learning Curve')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
