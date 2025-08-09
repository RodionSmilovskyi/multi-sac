import gym
import os
from datetime import datetime
import numpy as np
from typing import Final
from env.drone import FRAME_NUMBER, MAX_ALTITUDE, START_ALTITUDE, DroneEnv

height_range = [
    round(x, 4) for x in np.linspace(1.05 * START_ALTITUDE / MAX_ALTITUDE, 0.98, 10)
]
alt = 0.22
tolerance = 0.05
env = DroneEnv(True)
env.main_goal = np.array([0.88], dtype=np.float32)
start_pos = np.array([0, 0, 0.98], dtype=np.float32)

done = False
obs, info = env.reset(start_pos)
score = 0
frame = 0
max_steps = 600
while not done:
    action = env.action_space.sample()
    obs_, reward, terminated, truncated, info_ = env.step(action)
    done = terminated or truncated

    obs = obs_
    info = info_
    score += reward
    print(f"frame {frame} obs {obs} action {action} reward {reward}, vertical velocity {info['vertical_velocity']}")
    env.render()

    frame += 1

    if done and frame == FRAME_NUMBER:
        done = False
        
    if frame == max_steps:
        done = True

print(f"Main goal {env.main_goal}")
print(f"Total score {score}")
