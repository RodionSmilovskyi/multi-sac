import os
import sys
import pickle
import gymnasium as gym
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions.normal import Normal
from typing import Callable

from buffer import ReplayBuffer

# Assuming algos.buffer.ReplayBuffer is defined elsewhere


class CriticNetwork(nn.Module):
    """Critic network (Q-function). Unchanged from your original code."""

    def __init__(
        self,
        learning_rate: float,
        input_dims: tuple,
        n_actions: int,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        name: str = "critic",
        chkpt_dir="tmp/sac",
    ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{name}_sac")

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        """Forward pass"""
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        """Save state"""
        T.save(
            {"optimizer": self.optimizer.state_dict(), "model": self.state_dict()},
            self.checkpoint_file,
        )

    def load_checkpoint(self):
        """Load state"""
        checkpoint = T.load(self.checkpoint_file, weights_only=True)
        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class ActorNetwork(nn.Module):
    """Actor network (Policy). Unchanged from your original code."""

    def __init__(
        self,
        learning_rate: float,
        input_dims: tuple,
        max_action: float,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        n_actions: int = 2,
        name: str = "actor",
        chkpt_dir="tmp/sac",
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{name}_sac")
        self.max_action = max_action
        self.reparam_noise = 1e-6
        # Define bounds for log_std for numerical stability
        self.log_std_min = -20
        self.log_std_max = 2

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.log_sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        """Forward pass"""
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        log_sigma = self.log_sigma(prob)
        log_sigma = T.clamp(log_sigma, self.log_std_min, self.log_std_max)
        sigma = T.exp(log_sigma)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        """Take sample action"""
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()  # Reparameterization trick
        else:
            actions = probabilities.sample()

        # 1. Get the squashed action (in range [-1, 1])
        action_squashed = T.tanh(actions)

        # 2. Apply the correction to the SQUASHED action
        log_probs = probabilities.log_prob(actions)
        # The term inside the log is now guaranteed to be non-negative
        log_probs -= T.log(1 - action_squashed.pow(2) + self.reparam_noise) 

        # 3. Scale the squashed action to get the final action for the environment
        action = action_squashed * T.tensor(self.max_action).to(self.device)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        """Save state"""
        T.save(
            {"optimizer": self.optimizer.state_dict(), "model": self.state_dict()},
            self.checkpoint_file,
        )

    def load_checkpoint(self):
        """Load state"""
        checkpoint = T.load(self.checkpoint_file, weights_only=True)
        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class Agent:
    def __init__(
        self,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,  # Learning rate for the temperature parameter
        input_dims=[8],
        max_action: float = 1,
        gamma=0.99,
        n_actions=2,
        max_size=1000000,
        tau=0.005,
        batch_size=256,
        reward_scale=2,
        chkpt_dir: str = "",
        best_score=0,
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale  # This is the reward_scale
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.frame_idx = 0
        self.best_score = best_score
        self.best_eval_score = best_score
        self.checkpoint_file = os.path.join(chkpt_dir, "agent_sac")

        # --- Actor and Critic Setup ---
        self.actor = ActorNetwork(
            actor_lr,
            input_dims,
            n_actions=n_actions,
            max_action=max_action,
            chkpt_dir=chkpt_dir,
        )
        self.critic_1 = CriticNetwork(
            critic_lr,
            input_dims,
            n_actions,
            name="critic_1",
            chkpt_dir=chkpt_dir,
        )
        self.critic_2 = CriticNetwork(
            critic_lr,
            input_dims,
            n_actions,
            name="critic_2",
            chkpt_dir=chkpt_dir,
        )
        self.target_critic_1 = CriticNetwork(
            critic_lr,
            input_dims,
            n_actions,
            name="target_critic_1",
            chkpt_dir=chkpt_dir,
        )
        self.target_critic_2 = CriticNetwork(
            critic_lr,
            input_dims,
            n_actions,
            name="target_critic_2",
            chkpt_dir=chkpt_dir,
        )

        # --- Automatic Temperature (Alpha) Tuning Setup ---
        self.target_entropy = -T.tensor(n_actions, dtype=T.float32).to(self.device)
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def evaluate(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu, _ = self.actor.forward(state)
        action = T.tanh(mu)
        final_action = action * T.tensor(self.actor.max_action).to(self.actor.device)
        return final_action.cpu().detach().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # --- Update Target Critic Networks ---
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_dict = dict(critic_1_params)
        critic_2_dict = dict(critic_2_params)
        target_critic_1_dict = dict(target_critic_1_params)
        target_critic_2_dict = dict(target_critic_2_params)

        for name in critic_1_dict:
            critic_1_dict[name] = (
                tau * critic_1_dict[name].clone()
                + (1 - tau) * target_critic_1_dict[name].clone()
            )
        self.target_critic_1.load_state_dict(critic_1_dict)

        for name in critic_2_dict:
            critic_2_dict[name] = (
                tau * critic_2_dict[name].clone()
                + (1 - tau) * target_critic_2_dict[name].clone()
            )
        self.target_critic_2.load_state_dict(critic_2_dict)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        T.save(
            {
                "frame_idx": self.frame_idx,
                "best_score": self.best_score,
                "best_eval_score": self.best_eval_score,
                "memory": pickle.dumps(self.memory),
            },
            self.checkpoint_file,
        )

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

        checkpoint = T.load(self.checkpoint_file, weights_only=False)
        self.frame_idx = checkpoint["frame_idx"]
        self.best_score = checkpoint["best_score"]
        self.best_eval_score = checkpoint["best_eval_score"]
        self.memory = pickle.loads(checkpoint["memory"])

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)

        # --- Critic Update ---
        with T.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(
                state_, reparameterize=True
            )
            q1_target = self.target_critic_1.forward(state_, next_actions)
            q2_target = self.target_critic_2.forward(state_, next_actions)
            q_target_min = T.min(q1_target, q2_target).view(-1)

            # The key SAC target calculation
            q_target = self.scale * reward + self.gamma * (1 - done.float()) * (
                q_target_min - self.alpha * next_log_probs.view(-1)
            )

        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)

        critic_1_loss = F.mse_loss(q1, q_target)
        critic_2_loss = F.mse_loss(q2, q_target)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # --- Actor and Alpha Update ---
        # Freeze critic gradients for this part
        for p in self.critic_1.parameters():
            p.requires_grad = False
        for p in self.critic_2.parameters():
            p.requires_grad = False

        pi_actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)

        q1_pi = self.critic_1.forward(state, pi_actions)
        q2_pi = self.critic_2.forward(state, pi_actions)
        q_pi_min = T.min(q1_pi, q2_pi).view(-1)

        actor_loss = (self.alpha.detach() * log_probs - q_pi_min).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Unfreeze critic gradients
        for p in self.critic_1.parameters():
            p.requires_grad = True
        for p in self.critic_2.parameters():
            p.requires_grad = True

        # --- Alpha (Temperature) Update ---
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # --- Soft Update Target Networks ---
        self.update_network_parameters()


class ActorInferenceWrapper(nn.Module):
    """
    A wrapper for the ActorNetwork that creates an ONNX-exportable
    forward pass for deterministic inference.
    """

    def __init__(self, trained_actor: ActorNetwork):
        super().__init__()
        self.actor = trained_actor
        # Ensure the wrapper is on the same device as the actor
        self.device = trained_actor.device
        self.max_action = T.tensor(trained_actor.max_action).to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """
        This forward pass is designed for inference. It gets the deterministic
        mean from the actor and applies the Tanh squashing function.
        """
        # 1. Get the raw mean (mu) from the original actor's forward pass
        mu, _ = self.actor.forward(state)

        # 2. Apply the Tanh squashing function to clip the action to [-1, 1]
        action = T.tanh(mu)

        # 3. Scale by max_action (if necessary, often max_action is 1.0)
        action = action * self.max_action

        return action


class ActorCollector(mp.Process):
    def __init__(
        self,
        shared_actor,
        experience_queue,
        stop_event,
        worker_id,
        input_dims: tuple,
        max_action: float,
        n_actions: int,
        max_steps_per_episode: int,
        env_factory: Callable[[], gym.Env],
    ):
        super(ActorCollector, self).__init__()
        self.worker_id = worker_id
        self.experience_queue = experience_queue
        self.shared_actor = shared_actor
        self.stop_event = stop_event
        self.local_actor = ActorNetwork(3e-4, shared_actor.input_dims, max_action, n_actions=n_actions)
        self.env = env_factory()
        self.max_steps_per_episode = max_steps_per_episode

    def run(self):
        print(f"Worker-{self.worker_id} started.")
        while not self.stop_event.is_set():
            self.local_actor.load_state_dict(self.shared_actor.state_dict())
            state, _ = self.env.reset()
            done = False
            for _ in range(self.max_steps_per_episode):  # Max steps per episode
                if done or self.stop_event.is_set():
                    break
                state_tensor = T.tensor([state], dtype=T.float32)

                with T.no_grad():
                    action, _ = self.local_actor.sample_normal(state_tensor)

                action_np = action.cpu().numpy()[0]
                next_state, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated
                self.experience_queue.put((state, action_np, reward, next_state, done))
                state = next_state
        self.env.close()
        print(f"Worker-{self.worker_id} finished.")
