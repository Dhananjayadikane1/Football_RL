import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(striker_rewards, keeper_rewards, save_path=None, smooth_window=10):
    episodes = range(len(striker_rewards))
    plt.figure(figsize=(10, 6))

    # Optionally smooth the reward curves
    def smooth(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    if len(striker_rewards) >= smooth_window:
        plt.plot(smooth(striker_rewards, smooth_window), label="Striker Reward (Smoothed)", color='blue')
        plt.plot(smooth(keeper_rewards, smooth_window), label="Goalkeeper Reward (Smoothed)", color='green')
    else:
        plt.plot(episodes, striker_rewards, label="Striker Reward", color='blue')
        plt.plot(episodes, keeper_rewards, label="Goalkeeper Reward", color='green')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("📈 Learning Curve: Striker vs Goalkeeper", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[✓] Plot saved to {save_path}")
    else:
        plt.show()
