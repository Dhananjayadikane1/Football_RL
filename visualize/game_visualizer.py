import matplotlib.pyplot as plt
import numpy as np
def visualize_penalty(striker_x, keeper_x, result):
    goal_width = 7.32  # meters
    goal_height = 2.44  # not used directly here, but good for future 2D plots

    plt.figure(figsize=(8, 2.5))
    
    # Goal line
    plt.axhline(0, color='black', linewidth=2)
    plt.plot([0, goal_width], [0, 0], 'k-', linewidth=2)

    # Ball (Striker shot location)
    plt.plot(striker_x, 0, 'ro', markersize=10, label='Ball (Shot)')

    # Goalkeeper position
    plt.plot(keeper_x, 0, 'bs', markersize=12, label='Goalkeeper')

    # Goal posts
    plt.plot([0, 0], [0, 1], 'gray')
    plt.plot([goal_width, goal_width], [0, 1], 'gray')

    plt.xlim(-0.5, goal_width + 0.5)
    plt.ylim(-0.5, 1)
    plt.xticks(np.linspace(0, goal_width, 8))
    plt.yticks([])
    plt.title(f"⚽ Penalty Result: {result}", fontsize=14, color='green' if result == "GOAL" else 'red')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
