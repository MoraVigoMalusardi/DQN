import numpy as np
from typing import Tuple

class ReplayBuffer:
    """
    FIFO replay buffer
    Stores transitions (obs, action, reward, next_obs, done) 
    Has finite capacity
    Samples random batches with uniform probability
    """
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, obs, action: int, reward: float, next_obs, done: bool):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """ Sample a batch of transitions uniformly """
        # idx es un array de índices aleatorios
        idx = np.random.randint(0, self.size, size=batch_size) # tamano=batch_size, valores entre 0 y self.size-1
        return (
            self.obs_buf[idx],
            self.act_buf[idx],
            self.rew_buf[idx],
            self.next_obs_buf[idx],
            self.done_buf[idx],
        )

    def __len__(self):
        return self.size