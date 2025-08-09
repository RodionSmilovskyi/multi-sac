import gym
import math
import gym.spaces
import gym.spaces.utils
import gym.utils
import numpy as np
import numpy.typing as npt
from gym.core import ActType
from typing import Final

from drone_env.drone import (
    FRAME_NUMBER,
    MAX_ALTITUDE,
    MIN_ALTITUDE,
    START_ALTITUDE,
    TILT_LITMIT,
)

HOVER_ALTITUDE = 3
HOVER_HEIGHT_TOLERANCE_RATIO = 0.05

TARGET_ALT: Final[float] = round(HOVER_ALTITUDE / MAX_ALTITUDE, 4)


def compute_vector_distance(
    current_state: npt.NDArray, target_state: npt.NDArray, tolerance: float
) -> float:
    d = np.linalg.norm(current_state - target_state)

    if d <= tolerance:
        return 0
    else:
        return d


def compute_vector_distance_reward(
    current_state: npt.NDArray, target_state: npt.NDArray, tolerance: float
) -> float:
    d = np.linalg.norm(current_state - target_state)

    if d <= tolerance:
        return 1
    else:
        return -d


class SimpleHowerRewardWrapper(gym.Wrapper):
    """Class gives reward for just hovering"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.utils.flatten_space(
            gym.spaces.Dict(
                {
                    "altitude": gym.spaces.Box(0, MAX_ALTITUDE, shape=(1,)),
                    "roll": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                    "pitch": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                    "yaw": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                }
            )
        )
        self.reward_range = (-2 * TILT_LITMIT * FRAME_NUMBER, FRAME_NUMBER)

    def reset(self):
        obs, _ = self.env.reset()
        altitude = round(obs["altitude"], 4)
        height_difference = abs(altitude - TARGET_ALT)
        height_difference_ratio = round(height_difference / TARGET_ALT, 4)

        return (
            np.array(
                [height_difference_ratio, obs["roll"], obs["pitch"], obs["yaw"]],
                dtype=np.float32,
            ),
            {},
        )

    def step(self, action: ActType):
        obs, _, terminated, truncated, info = self.env.step(action)
        altitude = round(obs["altitude"], 4)
        height_difference = abs(altitude - TARGET_ALT)
        height_difference_ratio = round(height_difference / TARGET_ALT, 4)
        tilt_penalty = round(
            (abs(obs["roll"]) + abs(obs["pitch"]) + abs(obs["yaw"])) / math.pi, 4
        )

        is_too_far_from_target = height_difference_ratio >= 1

        if self.env.use_gui is True:
            if is_too_far_from_target is True:
                print("Too far from target height -> episode ended")

        reward = 1 - height_difference_ratio - tilt_penalty

        return (
            np.array(
                [height_difference_ratio, obs["roll"], obs["pitch"], obs["yaw"]],
                dtype=np.float32,
            ),
            round(reward, 4),
            terminated or is_too_far_from_target,
            truncated,
            info,
        )


class EuclidianRewardHowerWrapper(gym.Wrapper):
    """Class gives reward for just hovering"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.utils.flatten_space(
            gym.spaces.Dict(
                {
                    "altitude": gym.spaces.Box(0, 1, shape=(1,)),
                    "roll": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                    "pitch": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                    "yaw": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                }
            )
        )
        self.target_state: npt.NDArray = np.array([0.0, 0.0, 0.0, 0.0])
        self.tolerance: Final[float] = 0.05

        worst_state = self._compute_reward(
            np.array(
                [
                    1 - TARGET_ALT,
                    self.observation_space.high[1],
                    self.observation_space.high[2],
                    self.observation_space.high[3],
                ]
            ),
            self.target_state,
            self.tolerance,
        )

        self.reward_range = (worst_state * FRAME_NUMBER, FRAME_NUMBER)

    def reset(self):
        obs, _ = self.env.reset()
        altitude = round(obs["altitude"], 4)
        height_difference = altitude - TARGET_ALT

        return (
            np.array(
                [height_difference, obs["roll"], obs["pitch"], obs["yaw"]],
                dtype=np.float32,
            ),
            {},
        )

    def step(self, action: ActType):
        obs, _, terminated, truncated, info = self.env.step(action)
        altitude = round(obs["altitude"], 4)
        height_difference = altitude - TARGET_ALT
        current_state = np.array(
            [height_difference, obs["roll"], obs["pitch"], obs["yaw"]], dtype=np.float32
        )
        reward = self._compute_reward(current_state, self.target_state, self.tolerance)

        return (
            current_state,
            round(reward, 4),
            terminated,
            truncated,
            info,
        )

    def _compute_reward(
        self, current_state: npt.NDArray, target_state: npt.NDArray, tolerance: float
    ) -> float:
        d = np.linalg.norm(current_state - target_state)

        if d <= tolerance:
            return 1
        else:
            return -d


class ComplexHowerRewardWrapper(gym.Wrapper):
    """Class gives reward for just hovering"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.utils.flatten_space(
            gym.spaces.Dict(
                {
                    "altitude": gym.spaces.Box(0, MAX_ALTITUDE, shape=(1,)),
                    "roll": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                    "pitch": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                    "yaw": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                }
            )
        )
        self.reward_range = ((-1 - 3 * TILT_LITMIT**2) * FRAME_NUMBER, 0)

    def reset(self):
        obs, _ = self.env.reset()
        altitude = round(obs["altitude"], 4)
        height_difference = abs(altitude - TARGET_ALT)
        height_difference_ratio = round(height_difference / TARGET_ALT, 4)

        return (
            np.array(
                [height_difference_ratio, obs["roll"], obs["pitch"], obs["yaw"]],
                dtype=np.float32,
            ),
            {},
        )

    def step(self, action: ActType):
        altitude_weight = 1
        rotation_weight = 0.1

        obs, _, terminated, truncated, info = self.env.step(action)
        altitude = round(obs["altitude"], 4)
        height_difference = abs(altitude - TARGET_ALT)
        height_difference_ratio = round(height_difference / TARGET_ALT, 4)

        altitude_penalty = height_difference_ratio**2

        rotation_penalty = obs["roll"] ** 2 + obs["pitch"] ** 2 + obs["yaw"] ** 2

        is_too_far_from_target = height_difference_ratio >= 1

        if self.env.use_gui is True:
            if is_too_far_from_target is True:
                print("Too far from target height -> episode ended")

        reward = -(
            altitude_penalty * altitude_weight + rotation_weight * rotation_penalty
        )

        return (
            np.array(
                [height_difference_ratio, obs["roll"], obs["pitch"], obs["yaw"]],
                dtype=np.float32,
            ),
            round(reward, 4),
            terminated or is_too_far_from_target,
            truncated,
            info,
        )


class BaseWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.utils.flatten_space(
            gym.spaces.Dict(
                {
                    "altitude": gym.spaces.Box(0, 1, shape=(1,)),
                    "roll": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                    "pitch": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                    "yaw": gym.spaces.Box(-math.pi, math.pi, shape=(1,)),
                }
            )
        )

    def reset(self):
        obs, _ = self.env.reset()
        altitude = round(obs["altitude"], 4)
        height_difference = altitude - TARGET_ALT

        return (
            np.array(
                [height_difference, obs["roll"], obs["pitch"], obs["yaw"]],
                dtype=np.float32,
            ),
            {},
        )

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        altitude = round(obs["altitude"], 4)
        height_difference = altitude - TARGET_ALT
        current_state = np.array(
            [height_difference, obs["roll"], obs["pitch"], obs["yaw"]], dtype=np.float32
        )

        return (
            current_state,
            0,
            terminated,
            truncated,
            info,
        )


class InAirRewardWrapper(BaseWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.reward_range = (0, FRAME_NUMBER)

    def step(self, action):
        (
            current_state,
            _,
            terminated,
            truncated,
            info,
        ) = super().step(action)
        
        in_air_reward = 1
        
        return (
            current_state,
            in_air_reward,
            terminated,
            truncated,
            info,
        )


class StableFlightWrapper(InAirRewardWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        worst_state = compute_vector_distance_reward(
            np.array(
                [
                    self.observation_space.high[1],
                    self.observation_space.high[2],
                    self.observation_space.high[3],
                ]
            ),
            np.array([0., 0., 0.], dtype=np.float32),
            0.05,
        )

        self.reward_range = (worst_state * FRAME_NUMBER, 2 * FRAME_NUMBER)

    def step(self, action):
        (
            current_state,
            in_air_reward,
            terminated,
            truncated,
            info,
        ) = super().step(action)
        _, roll, pitch, yaw = current_state

        stable_flight_reward = compute_vector_distance_reward(
            np.array([roll, pitch, yaw], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            0.05,
        )

        return (
            current_state,
            stable_flight_reward + in_air_reward,
            terminated,
            truncated,
            info,
        )


class HoverAltWrapper(StableFlightWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        worst_state = compute_vector_distance_reward(
            np.array(
                [
                    1 - TARGET_ALT,
                    self.observation_space.high[1],
                    self.observation_space.high[2],
                    self.observation_space.high[3],
                ]
            ),
            np.array([0., 0., 0., 0.], dtype=np.float32),
            0.05,
        )

        self.reward_range = (worst_state * FRAME_NUMBER, FRAME_NUMBER)
        self.reward_range = (0, 3 * FRAME_NUMBER)

    def step(self, action):
        (
            current_state,
            stable_flight_reward,
            terminated,
            truncated,
            info,
        ) = super().step(action)

        height_difference, roll, pitch, yaw = current_state

        height_difference_reward = compute_vector_distance_reward(
            np.array([height_difference], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            0.05,
        )

        return (
            current_state,
            height_difference_reward + stable_flight_reward,
            terminated,
            truncated,
            info,
        )

