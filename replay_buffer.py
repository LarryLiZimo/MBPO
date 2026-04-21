import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
                  next_obs: np.ndarray, done: np.ndarray):
        batch_size = obs.shape[0]
        end_ptr = self.ptr + batch_size

        if end_ptr <= self.capacity:
            self.obs[self.ptr:end_ptr] = obs
            self.action[self.ptr:end_ptr] = action
            self.reward[self.ptr:end_ptr] = reward
            self.next_obs[self.ptr:end_ptr] = next_obs
            self.done[self.ptr:end_ptr] = done
        else:
            overflow = end_ptr - self.capacity
            first_chunk = batch_size - overflow
            self.obs[self.ptr:] = obs[:first_chunk]
            self.action[self.ptr:] = action[:first_chunk]
            self.reward[self.ptr:] = reward[:first_chunk]
            self.next_obs[self.ptr:] = next_obs[:first_chunk]
            self.done[self.ptr:] = done[:first_chunk]
            self.obs[:overflow] = obs[first_chunk:]
            self.action[:overflow] = action[first_chunk:]
            self.reward[:overflow] = reward[first_chunk:]
            self.next_obs[:overflow] = next_obs[first_chunk:]
            self.done[:overflow] = done[first_chunk:]

        self.ptr = end_ptr % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.obs[idx], self.action[idx], self.reward[idx], self.next_obs[idx], self.done[idx]

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def clear(self):
        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return self.size
