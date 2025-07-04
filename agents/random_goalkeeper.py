import random

class RandomGoalkeeperAgent:
    def __init__(self):
        self.actions = [0, 1, 2]  # 0: left, 1: stay, 2: right

    def choose_action(self, state):
        return random.choice(self.actions), 1.0  # return dummy prob for consistency
