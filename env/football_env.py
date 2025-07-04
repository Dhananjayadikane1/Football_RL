import time
import numpy as np
import math

class FootballEnv:
    def __init__(self):
        self.goal_width = 7.32  # Standard goal width in meters
        self.goal_center = self.goal_width / 2

        self.ball_start_pos = np.array([self.goal_center, 11.0])  # Penalty spot: 11m away
        self.keeper_start_x = self.goal_center

        self.keeper_speed = 1.0  # per step
        self.response_delay = 0.02  # seconds (reaction delay)

        self.reset()

    def reset(self):
        self.ball_pos = self.ball_start_pos.copy()
        self.keeper_x = self.keeper_start_x
        return self.get_state()

    def get_state(self):
        # shot angle: from ball to center of goal
        shot_angle = self.calculate_shot_angle(self.ball_pos, self.goal_center)
        return np.array([
            self.keeper_x,              # Goalkeeper X position
            self.ball_pos[0],           # Ball X position
            self.ball_pos[1],           # Ball Y position
            self.goal_center,           # Goal center X
            shot_angle                  # Angle from ball to goal center
        ], dtype=np.float32)

    def step(self, striker_target_x, keeper_action):
        # Simulate striker shot
        ball_target_x = striker_target_x

        # Keeper moves (3 actions: left, stay, right)
        if keeper_action == 0:
            self.keeper_x -= self.keeper_speed
        elif keeper_action == 2:
            self.keeper_x += self.keeper_speed

        self.keeper_x = np.clip(self.keeper_x, 0, self.goal_width)

        # Ball reaches goal line
        self.ball_pos[1] = 0.0
        self.ball_pos[0] = ball_target_x

     # Determine outcome
        result = self._check_goal_or_save(self.keeper_x, self.ball_pos[0])
        reward_striker = 1 if result == "GOAL" else -1
        reward_keeper = -reward_striker

        done = True
        next_state = self.get_state()

        return next_state, reward_striker, reward_keeper, result, done  # ✅ exactly 5 values


    def _check_goal_or_save(self, keeper_x, shot_x):
        # If keeper is within 0.5 meters of shot, it's a SAVE
        if abs(keeper_x - shot_x) <= 0.5:
            return "SAVE"
        return "GOAL"

    def calculate_shot_angle(self, ball_pos, target_x):
        dx = target_x - ball_pos[0]
        dy = 0 - ball_pos[1]
        return math.atan2(dy, dx)  # in radians
