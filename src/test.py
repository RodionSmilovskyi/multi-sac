import gymnasium as gym
import os
from datetime import datetime
import numpy as np
from typing import Final,Callable

import torch as T
import torch.multiprocessing as mp
import time

from buffer import ReplayBuffer
from networks import ActorCollector, ActorNetwork, Agent

def evaluate_agent(
    agent: Agent,
    env_factory: Callable[[], gym.Env],
    max_frames: int,
    number_of_iterations: int = 1,
) -> tuple[float, float]:
    """Evaluate agent, using deterministic evaluate function"""
    total_score = 0
    total_length = 0
    env = env_factory()
    
    for ind in range(number_of_iterations):
        frame = 0
        done = False
        obs, _ = env.reset()

        while not done:
            action = agent.evaluate(obs)

            obs_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = obs_

            if frame == max_frames:
                done = True

            total_score += reward
            total_length += 1
            frame += 1

    return (
        round(total_score / number_of_iterations, 2),
        round(total_length / number_of_iterations, 2),
    )

if __name__ == "__main__":
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print(f"Learner using device: {device}")
    
    INPUT_DIMS = 3       # [altitude, roll, pitch, yaw_velocity]
    N_ACTIONS = 2        # [throttle, roll, pitch, yaw]
    ACTION_BOUND_LOW = -2.0
    ACTION_BOUND_HIGH = 2.0
    NUM_WORKERS = mp.cpu_count()
    REPLAY_BUFFER_CAPACITY = 1_000_000
    TOTAL_TRAINING_STEPS = 500_000
    LEARNING_STARTS_AFTER = 10000
    SAVE_INTERVAL = 50000
    EVAL_INTERVAL = 25000

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    dummy_env = gym.make("Pendulum-v1")
    shared_actor = ActorNetwork(3e-4, dummy_env.observation_space.shape, max_action=ACTION_BOUND_HIGH, n_actions=N_ACTIONS)
    shared_actor.share_memory()
    experience_queue = mp.Queue()
    stop_event = mp.Event()
    
    learner_agent = Agent(input_dims=dummy_env.observation_space.shape, max_action=ACTION_BOUND_HIGH, n_actions=N_ACTIONS)
    best_eval_reward = -float('inf')
    steps_collected = 0
    next_eval_step = steps_collected + (EVAL_INTERVAL - (steps_collected % EVAL_INTERVAL))

    workers = [ActorCollector(shared_actor, experience_queue, stop_event, i, dummy_env.observation_space.shape, ACTION_BOUND_HIGH, N_ACTIONS, 200, lambda: gym.make("Pendulum-v1")) for i in range(NUM_WORKERS)]
    [w.start() for w in workers]
    
    while best_eval_reward < -10:
        while not experience_queue.empty():
            state, action_np, reward, next_state, done = experience_queue.get()
            learner_agent.memory.store_transition(state, action_np, reward, next_state, done) 
            steps_collected += 1
            
        if steps_collected > LEARNING_STARTS_AFTER: 
            learner_agent.learn()
            
        shared_actor.load_state_dict(learner_agent.actor.state_dict())
        if steps_collected >= next_eval_step:
            avg_reward, _ = evaluate_agent(learner_agent, lambda: gym.make("Pendulum-v1"), 200, 5)
            print(f"Evaluation reward: {avg_reward:.2f}.")
            if avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                print(f"New best evaluation reward: {best_eval_reward:.2f}.")

            next_eval_step += EVAL_INTERVAL
            
        time.sleep(0.001)

# from env.drone import FRAME_NUMBER, MAX_ALTITUDE, START_ALTITUDE, DroneEnv

# height_range = [
#     round(x, 4) for x in np.linspace(1.05 * START_ALTITUDE / MAX_ALTITUDE, 0.98, 10)
# ]
# alt = 0.22
# tolerance = 0.05
# env = DroneEnv(True)
# env.main_goal = np.array([0.88], dtype=np.float32)
# start_pos = np.array([0, 0, 0.98], dtype=np.float32)

# done = False
# obs, info = env.reset(start_pos)
# score = 0
# frame = 0
# max_steps = 600
# while not done:
#     action = env.action_space.sample()
#     obs_, reward, terminated, truncated, info_ = env.step(action)
#     done = terminated or truncated

#     obs = obs_
#     info = info_
#     score += reward
#     print(f"frame {frame} obs {obs} action {action} reward {reward}, vertical velocity {info['vertical_velocity']}")
#     env.render()

#     frame += 1

#     if done and frame == FRAME_NUMBER:
#         done = False

#     if frame == max_steps:
#         done = True

# print(f"Main goal {env.main_goal}")
# print(f"Total score {score}")
