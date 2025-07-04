import sys
import os
import numpy as np
import torch
from env.football_env import FootballEnv
from agents.dqn_striker import DQNStrikerAgent
from agents.ppo_goalkeeper import PPOGoalkeeperAgent
from visualize.game_visualizer import visualize_penalty

def main():
    print("⚽ Loading Environment and Agents...")
    env = FootballEnv()
    state_dim = 5
    striker_action_dim = 10
    keeper_action_dim = 3

    # Load trained agents (or new instances if not trained)
    striker = DQNStrikerAgent(state_dim, striker_action_dim)
    keeper = PPOGoalkeeperAgent(state_dim, keeper_action_dim)

    # Example: Simulate 10 sequential penalties
    num_penalties = 10
    results = []

    print(f"🎮 Simulating {num_penalties} penalty shots...\n")
    for i in range(num_penalties):
        state = env.reset()

        # Striker decides where to shoot
        striker_action_idx = striker.choose_action(state)
        striker_target_x = np.linspace(0.5, 6.8, striker_action_dim)[striker_action_idx]

        # Goalkeeper decides where to dive
        keeper_action_idx, _ = keeper.choose_action(state)

        # Environment executes
        next_state, reward_striker, reward_keeper, result, done = env.step(striker_target_x, keeper_action_idx)

        # Record result
        print(f"[Penalty {i+1}] Striker aimed at x={striker_target_x:.2f}, Keeper moved to x={next_state[0]:.2f} => {result}")
        visualize_penalty(striker_target_x, next_state[0], result)
        results.append(result)

    # Summary
    goals = results.count("GOAL")
    saves = results.count("SAVE")
    print(f"\n✅ Summary after {num_penalties} penalties: {goals} GOAL(s), {saves} SAVE(s)")

if __name__ == "__main__":
    main()
