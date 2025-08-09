import numpy as np
import numpy.typing as npt
import torch as T

from algos.sac.networks import ActorNetwork, Agent
from drone_env.uvfa_wrappers import UVFAWrapper


# --- PID Controller Class ---
class PIDController:
    """A simple PID controller."""

    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.last_error, self.integral = 0, 0

    def reset(self):
        self.last_error, self.integral = 0, 0

    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


# --- Expert Controller ---
class ExpertController:
    """
    A cascaded PID controller that acts as an expert policy to generate
    demonstration data for reaching and holding a target altitude.
    """

    def __init__(self, target_altitude, hover_throttle):
        self.target_altitude = target_altitude
        self.hover_throttle = hover_throttle

        # --- 1. Outer Loop (Position -> Desired Velocity) ---
        # This is a P-controller. It decides how fast to move.
        # Gains are tuned manually to produce smooth, stable ascents.
        self.pos_p_gain = 2.0
        self.max_vertical_speed = 2  # Cap the desired speed

        # --- 2. Inner Loop (Velocity -> Throttle Adjustment) ---
        # This is a full PID controller to achieve the desired velocity.
        self.vel_pid = PIDController(Kp=1, Ki=0, Kd=0)

    def get_action(self, current_altitude, current_vertical_velocity):
        """
        Calculates the 4D action vector based on the current state.

        Args:
            observation (np.array): The state from the environment.

        Returns:
            np.array: The 4D action [roll, pitch, yaw_rate, throttle].
        """
        # --- Outer Loop Calculation ---
        altitude_error = self.target_altitude - current_altitude
        desired_velocity = self.pos_p_gain * altitude_error
        # Clamp the desired velocity to a maximum
        desired_velocity = np.clip(
            desired_velocity, -self.max_vertical_speed, self.max_vertical_speed
        )

        # --- Inner Loop Calculation ---
        self.vel_pid.setpoint = desired_velocity
        throttle_adjustment = self.vel_pid.compute(
            current_vertical_velocity, dt=1.0 / 240.0
        )

        # --- Final Action ---
        # The final throttle is the base hover throttle plus the adjustment
        final_throttle = self.hover_throttle + throttle_adjustment
        final_throttle = np.clip(final_throttle, -1.0, 1.0)

        # For this task, the expert always wants to be perfectly stable
        target_roll = 0.0
        target_pitch = 0.0
        target_yaw_rate = 0.0

        return np.array([final_throttle, target_roll, target_pitch, target_yaw_rate])

    @classmethod
    def warmup_from_expert(
        cls,
        agent: Agent,
        env: UVFAWrapper,
        max_frames_per_episode: int,
        total_number_of_frames: int,
        init_state: npt.NDArray,
        goal: npt.NDArray,
        hover_throttle: float,
        warmup_frames: int = 0,
        min_student_accuracy: float = 0.005,
    ):

        cls.populate_memory_buffer(
            agent,
            env,
            max_frames_per_episode,
            total_number_of_frames,
            init_state,
            goal,
            hover_throttle,
        )

        student_network = ActorNetwork(
            0.0001,
            env.observation_space.shape,
            n_actions=env.action_space.shape[0],
            max_action=env.action_space.high.max(),
            chkpt_dir=agent.actor.checkpoint_dir,
            name=f"student_dist",
        )

        dist_loss_history = []
        avg_loss = None
        it = 0

        while avg_loss is None or avg_loss > min_student_accuracy:
            state, action, reward, new_state, done = agent.memory.sample_buffer(1024)
            state = T.tensor(state, dtype=T.float).to(student_network.device)
            expert_action = T.tensor(action, dtype=T.float).to(student_network.device)

            mu, sigma = student_network(state)
            student_action = T.tanh(mu) 
            loss = T.nn.functional.mse_loss(student_action, expert_action)

            student_network.optimizer.zero_grad()
            loss.backward()
            student_network.optimizer.step()

            dist_loss_history.append(loss.item())
            avg_loss = np.mean(dist_loss_history[-10:])

            if it % 100 == 0:
                print(f"Iteration {it} loss {loss} avg_loss {avg_loss}")

            it += 1
            
        student_state_dict = student_network.state_dict()
        agent.actor.load_state_dict(student_state_dict)
        print("Actor warmed up")
        
        for _ in range(warmup_frames):
            agent.learn()
            
        print("Agent warmed up")
        

    @classmethod
    def populate_memory_buffer(
        cls,
        agent: Agent,
        env: UVFAWrapper,
        max_frames_per_episode: int,
        total_number_of_frames: int,
        init_state: npt.NDArray,
        goal: npt.NDArray,
        hover_throttle: float,
    ):
        """Populate env with expert results"""
        frame_count = 0
        env.main_goal = goal
        expert = cls(env.main_goal[0], hover_throttle)

        while frame_count < total_number_of_frames:
            done = False
            obs, info = env.reset(init_state)
            current_episode_frame = 0
            expert.vel_pid.reset()

            while not done:
                action = expert.get_action(obs[0], info["vertical_velocity"])
                obs_, reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated

                agent.remember(obs, action, reward, obs_, done)
                obs = obs_
                info = _info
                current_episode_frame += 1
                frame_count += 1

                if current_episode_frame >= max_frames_per_episode:
                    done = True

    @classmethod
    def get_expert_experience(
        cls,
        env: UVFAWrapper,
        max_frames_per_episode: int,
        total_number_of_frames: int,
        init_state: npt.NDArray,
        goal: npt.NDArray,
        hover_throttle: float,
    ):
        frame_count = 0
        env.main_goal = goal
        expert = cls(env.main_goal[0], hover_throttle)
        result = []
        while frame_count < total_number_of_frames:
            done = False
            obs, info = env.reset(init_state)
            current_episode_frame = 0
            expert.vel_pid.reset()

            while not done:
                action = expert.get_action(obs[0], info["vertical_velocity"])
                obs_, reward, terminated, truncated, info_ = env.step(action)
                done = terminated or truncated
                result.append((obs, action, reward, obs_, done))
                obs = obs_
                info = info_

                current_episode_frame += 1
                frame_count += 1

                if current_episode_frame >= max_frames_per_episode:
                    done = True

        return result
