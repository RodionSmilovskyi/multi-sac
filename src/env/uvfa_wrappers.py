# pylint: disable=C0114, C0413, E0401, W0621
import math
import random
from typing import TypeAlias
import gym
import numpy as np
import numpy.typing as npt

from gym.core import ObsType, ActType
from drone_env.drone import FRAME_NUMBER, MAX_ALTITUDE, START_ALTITUDE
from drone_env.wrappers import TARGET_ALT

Trajectory: TypeAlias = tuple[ObsType, ActType, float, ObsType, bool]


def _compute_vector_distance_reward(
    current_state: npt.NDArray, target_state: npt.NDArray, tolerance: float
) -> float:
    """Reward based on vector distance between states
    from -distance to 1 if less then tolerance
    """
    d = np.linalg.norm(current_state - target_state)

    if d <= tolerance:
        return 1
    else:
        return -d


def _split_state_goal(
    state_goal: npt.NDArray, mid_point: int
) -> tuple[npt.NDArray, npt.NDArray]:
    # Split the array
    state = state_goal[:mid_point]
    goal = state_goal[mid_point:]

    return state, goal


def _adjust_for_new_goal(
    old_state_goal: npt.NDArray, old_next_state_goal: npt.NDArray, new_goal: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, float]:
    """HER goal adjustment"""
    old_state, _ = _split_state_goal(old_state_goal, 4)
    old_next_state, _ = _split_state_goal(old_next_state_goal, 4)

    flight_reward = _compute_vector_distance_reward(old_next_state, new_goal, 0.1)
    in_air_reward = 1
    new_state_goal = np.concatenate((old_state, new_goal))
    new_next_state_goal = np.concatenate((old_next_state, new_goal))

    return (
        new_state_goal,
        new_next_state_goal,
        (0.5 * in_air_reward + 0.5 * flight_reward),
    )


def random_goal_strategy(trajectories: list[Trajectory]) -> list[Trajectory]:
    """Assigns random goal to a list of trajectories"""
    result: list[Trajectory] = []
    random_goal = np.array(
        [
            round(np.random.uniform(START_ALTITUDE / MAX_ALTITUDE, 0.9), 4),
            round(np.random.uniform(-math.pi, math.pi), 4),
            round(np.random.uniform(-math.pi, math.pi), 4),
            round(np.random.uniform(-math.pi, math.pi), 4),
        ]
    )

    for observation, action, _, observation_, done in trajectories:
        o, o_, r_ = _adjust_for_new_goal(observation, observation_, random_goal)
        result.append((o, action, r_, o_, done))

    return result


def final_goal_strategy(trajectories: list[Trajectory]) -> list[Trajectory]:
    """Implement startegy, where final state of each episode is an additional goal"""

    final_step, _ = _split_state_goal(trajectories[-1][3], 4)
    result: list[Trajectory] = []

    for observation, action, _, observation_, done in trajectories:
        o, o_, r_ = _adjust_for_new_goal(observation, observation_, final_step)
        result.append((o, action, r_, o_, done))

    return result


def future_goal_strategy(
    trajectories: list[Trajectory], future_steps: int
) -> list[Trajectory]:
    """Addition goal startegy based on future steps"""
    trajectory_length = len(trajectories)
    result: list[Trajectory] = []

    for ind, t in enumerate(trajectories):
        number_of_samples = min(future_steps, trajectory_length - 1 - ind)
        selected_trajectories = random.sample(
            trajectories[ind + 1 :], number_of_samples
        )
        observation, action, _, observation_, done = t

        for st in selected_trajectories:
            goal, _ = _split_state_goal(st[3], 4)
            o, o_, r_ = _adjust_for_new_goal(observation, observation_, goal)
            result.append((o, action, r_, o_, done))

    return result


class UVFAWrapper(gym.Wrapper):
    """Wrapper which uses Universal Value Function Approximators for training"""

    def __init__(self, env: gym.Env, tolerance: float = 0.1):
        super().__init__(env)

        self.observation_space = gym.spaces.utils.flatten_space(
            gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "altitude": gym.spaces.Box(0, 1, shape=(1,)),
                            "roll": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                            "pitch": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                            "yaw": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                        }
                    ),
                    "goal": gym.spaces.Dict(
                        {"altitude": gym.spaces.Box(0, 1, shape=(1,))}
                    ),
                }
            )
        )

        self.main_goal = np.array([TARGET_ALT], dtype=np.float32)
        worst_state = _compute_vector_distance_reward(
            np.array([self.observation_space.high[0]]),
            self.main_goal,
            0.1,
        )

        self.reward_range = (worst_state * FRAME_NUMBER, FRAME_NUMBER)
        self.tolerance = tolerance

    def reset(self, start_pos=None) -> tuple[ObsType, dict]:
        obs, info = (
            self.env.reset([start_pos[0], start_pos[1], start_pos[2] * MAX_ALTITUDE])
            if start_pos is not None
            else self.env.reset()
        )

        current_state = np.array(
            [
                round(obs["altitude"], 4),
                round(obs["roll"], 4),
                round(obs["pitch"], 4),
                round(obs["yaw"], 4),
            ],
            dtype=np.float32,
        )
        state_goal = np.concatenate((current_state, self.main_goal))

        return (
            state_goal,
            info,
        )

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        current_state = np.array(
            [
                round(obs["altitude"], 4),
                round(obs["roll"], 4),
                round(obs["pitch"], 4),
                round(obs["yaw"], 4),
            ],
            dtype=np.float32,
        )
        state_goal = np.concatenate((current_state, self.main_goal))
        reward = self.reward_function(state_goal, info)

        return (
            state_goal,
            reward,
            terminated,
            truncated,
            info,
        )

    def reward_function(self, state_goal: npt.NDArray, info: dict = {}) -> float:
        """Basic reward function which takes vector distance between current state and goal"""
        state, goal = _split_state_goal(state_goal, 4)
        in_air_reward = 1
        no_tilt_reward = _compute_vector_distance_reward(
            np.array([state[1], state[2], state[3]], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            self.tolerance,
        )
        height_reward = _compute_vector_distance_reward(
            np.array([state[0]], dtype=np.float32),
            np.array([goal[0]], dtype=np.float32),
            self.tolerance,
        )

        velocity_penalty = (
            -0.1 * np.abs(info["vertical_velocity"])
            if "vertical_velocity" in info
            else 0
        )

        return (
            0.5 * height_reward
            + 0.3 * no_tilt_reward
            + 0.2 * in_air_reward
            + velocity_penalty
        )


class InAirUVFAWrapper(UVFAWrapper):
    def reset(self, start_pos=None) -> tuple[ObsType, dict]:
        self.main_goal = np.array(
            [random.uniform(START_ALTITUDE / MAX_ALTITUDE, 1)], dtype=np.float32
        )

        return super().reset()

    def reward_function(self, state_goal):
        return 1


class StableFlightUFAWrapper(UVFAWrapper):
    def __init__(self, env: gym.Env, tolerance: float = 0.1):
        super().__init__(env, tolerance)
        self.reward_range = (-500, 500)

    def reward_function(self, state_goal):
        state, goal = _split_state_goal(state_goal, 4)
        in_air_reward = 1
        no_tilt_penalty = round(
            np.linalg.norm(
                np.array([state[1], state[2], state[3]], dtype=np.float32)
                - np.array([0, 0, 0], dtype=np.float32)
            ),
            2,
        )

        if no_tilt_penalty < self.tolerance:
            no_tilt_penalty = 0

        return in_air_reward - no_tilt_penalty


class MultiTargetUVFAWrapper(UVFAWrapper):
    """Class which support picking random target from the list"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.targets: list[npt.NDArray] = []

    def reset(self, **kwargs) -> tuple[ObsType, dict]:
        self.main_goal = random.choice(self.targets)

        return super().reset()


class RandomStartUVFAWrapper(UVFAWrapper):
    def __init__(self, env: gym.Env, tolerance: float = 0.1):
        super().__init__(env, tolerance)
        self.start_pos: list[npt.NDArray] = [np.array([0, 0, 0], dtype=np.float32)]

    def reset(self) -> tuple[ObsType, dict]:
        start_pos = random.choice(self.start_pos)
        return super().reset([start_pos[0], start_pos[1], start_pos[2] * MAX_ALTITUDE])


class AdvancedControl(UVFAWrapper):
    def _compute_vector_distance_reward(
        self, current_state: npt.NDArray, target_state: npt.NDArray, tolerance: float
    ) -> float:
        """
        Calculates a smooth reward based on the distance to a target.
        The reward is 1 at the target and decreases smoothly towards 0.
        """
        distance = np.linalg.norm(current_state - target_state)

        # 'k' is a tuning parameter that controls how "sharp" the reward peak is.
        # A smaller 'k' creates a wider, more gentle hill for the agent to climb.
        # Let's start with a value that gives a decent reward within the tolerance.
        k = 100

        reward = np.exp(-k * (distance**2))

        return reward
    
    def __init__(self, env: gym.Env, tolerance: float = 0.1):
        self.current_action = np.zeros(4)
        self.prev_action = np.zeros(4)
        super().__init__(env, tolerance)
        
    def step(self, action):
        self.current_action = action
        result = super().step(action)
        self.prev_action = action
        return result

    def reset(self, start_pos) -> tuple[ObsType, dict]:
        self.current_action = np.zeros(4)
        self.prev_action = np.zeros(4)
        return super().reset(start_pos)
    
    def reward_function(self, state_goal: npt.NDArray, info: dict = {}) -> float:
        """Basic reward function which takes vector distance between current state and goal"""
        state, goal = _split_state_goal(state_goal, 4)
        time_penalty = -0.01
        # no_tilt_reward = self._compute_vector_distance_reward(
        #     np.array([state[1], state[2], state[3]], dtype=np.float32),
        #     np.array([0, 0, 0], dtype=np.float32),
        #     self.tolerance,
        # )
        # height_reward = self._compute_vector_distance_reward(
        #     np.array([state[0]], dtype=np.float32),
        #     np.array([goal[0]], dtype=np.float32),
        #     self.tolerance,
        # )

        # velocity_penalty = (
        #     -0.3 * (info["vertical_velocity"] ** 2)
        #     if "vertical_velocity" in info
        #     else 0
        # )

        # distance_to_target = np.linalg.norm(
        #     np.array([state[0]], dtype=np.float32)
        #     - np.array([goal[0]], dtype=np.float32)
        # )

        # arrival_bonus = 0

        # if (
        #     distance_to_target < self.tolerance
        #     and np.abs(info["vertical_velocity"]) < 0.1
        # ):
        #     arrival_bonus = 10.0  # Large bonus for a slow, stable arrival
            
        # action_delta = self.current_action - self.prev_action
        # action_rate_penalty_k = 0.05
        # action_rate_penalty = -action_rate_penalty_k * np.sum(action_delta**2)

        # reward = (
        #     (0.6 * height_reward)
        #     + (0.4 * no_tilt_reward)
        #     + action_rate_penalty
        #     + time_penalty
        #     + velocity_penalty
        #     + arrival_bonus
        # )
        k = 100
        height_diff = np.abs(state[0] - goal[0])
        height_reward = np.exp(-k * (height_diff**2))
        time_penalty = -0.01

        return height_reward + time_penalty

    def calculate_success_threshold(
        self,
        distance: float,
        total_frames: int = 600,
        # --- Agent Behavior Parameters ---
        max_measured_velocity: float = 9.08,
        assumed_efficiency_factor: float = 0.25,
        # --- Reward Function Parameters ---
        height_reward_weight: float = 0.6,
        tilt_reward_weight: float = 0.4,
        velocity_penalty_k: float = 0.3,
        time_penalty: float = -0.01,
        arrival_bonus: float = 10.0,
        exp_reward_k: float = 10.0,
        # --- Simulation Parameters ---
        fps: int = 240,
    ) -> float:
        """
        Estimates a success score based on a complex, shaped reward function.

        Args:
            distance: The vertical distance the drone needs to travel (m).
            total_frames: The total number of frames in an episode.
            max_measured_velocity: The drone's measured max speed (m/s).
            assumed_efficiency_factor: The fraction of max speed a good agent uses.
            reward_function_params...: The coefficients from your reward function.
            fps: The frames per second of the simulation.

        Returns:
            An estimated success threshold score.
        """
        # 1. Calculate realistic travel time and frame distribution
        realistic_avg_velocity = max_measured_velocity * assumed_efficiency_factor
        travel_time_seconds = distance / realistic_avg_velocity
        travel_frames = min(travel_time_seconds * fps, total_frames)
        hover_frames = max(0, total_frames - travel_frames)

        # 2. Estimate average reward per frame during the TRAVEL phase
        # Agent is far from the goal, but we estimate the reward at the halfway point
        avg_travel_dist = distance / 2.0
        avg_height_reward = np.exp(-exp_reward_k * (avg_travel_dist**2))
        # Assume agent is mostly stable during travel
        avg_tilt_reward = 0.95
        # Velocity penalty is now squared
        avg_velocity_penalty = -velocity_penalty_k * (realistic_avg_velocity**2)

        reward_per_travel_frame = (
            (height_reward_weight * avg_height_reward)
            + (tilt_reward_weight * avg_tilt_reward)
            + avg_velocity_penalty
            + time_penalty
        )

        # 3. Estimate average reward per frame during the HOVER phase
        # Agent is very close to the goal and stable
        avg_hover_height_reward = 0.98  # Almost perfect
        avg_hover_tilt_reward = 0.98  # Almost perfect
        avg_hover_velocity_penalty = 0  # Almost still

        reward_per_hover_frame = (
            (height_reward_weight * avg_hover_height_reward)
            + (tilt_reward_weight * avg_hover_tilt_reward)
            + avg_hover_velocity_penalty
            + time_penalty
        )

        # 4. Calculate total score
        travel_score = travel_frames * reward_per_travel_frame
        hover_score = hover_frames * reward_per_hover_frame

        # The arrival bonus is a one-time event added at the end
        total_score = travel_score + hover_score + arrival_bonus

        return total_score


def height_goal_strategy(
    trajectories: list[Trajectory], env: UVFAWrapper
) -> list[Trajectory]:
    result: list[Trajectory] = []
    height_points = np.linspace(START_ALTITUDE / MAX_ALTITUDE, 1, 10)
    for observation, action, _, observation_, done in trajectories:
        state, _ = _split_state_goal(observation, 4)
        next_state, _ = _split_state_goal(observation_, 4)
        for hp in height_points:
            new_goal = np.array(
                [hp, next_state[1], next_state[2], next_state[3]], dtype=np.float32
            )
            new_state_goal = np.concatenate((state, new_goal))
            new_next_state_goal = np.concatenate((next_state, new_goal))
            new_reward = env.reward_function(new_next_state_goal)
            result.append(
                (new_state_goal, action, new_reward, new_next_state_goal, done)
            )

    return result


def angle_goal_strategy(trajectories: list[Trajectory]) -> list[Trajectory]:
    return []
