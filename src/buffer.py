"""
Buffer system for the RL
"""

import random
from collections import deque

import numpy as np

from common_definitions import BUFFER_UNBALANCE_GAP


class ReplayBuffer:
    """
    Replay Buffer to store the experiences.
    """

    def __init__(self, buffer_size, batch_size):
        """
        Initialize the attributes.

        Args:
            buffer_size: The size of the buffer memory
            batch_size: The batch for each of the data request `get_batch`
        """
        self.buffer = deque(maxlen=int(buffer_size))  # with format of (s,a,r,s')

        # constant sizes to use
        self.batch_size = batch_size

        # temp variables
        self.p_indices = [BUFFER_UNBALANCE_GAP/2]

    def append(self, state, action, reward, next_state, done):  # pylint: disable=too-many-arguments
        """
        Append to the Buffer

        Args:
            state: the state
            action: the action
            r: the reward
            sn: the next state
            d: done (whether one loop is done or not)
        """
        self.buffer.append([
            state, action, np.expand_dims(reward, -1),
            next_state, np.expand_dims(done, -1)
        ])

    def get_batch(self, unbalance_p=True):
        """
        Get the batch randomly from the buffer

        Args:
            unbalance_p: If true, unbalance probability of taking the batch from buffer with
            recent event being more prioritized

        Returns:
            the resulting batch
        """
        # unbalance indices
        p_indices = None
        if random.random() < unbalance_p:
            self.p_indices.extend((np.arange(len(self.buffer)-len(self.p_indices))+1)
                                  * BUFFER_UNBALANCE_GAP + self.p_indices[-1])
            p_indices = self.p_indices / np.sum(self.p_indices)

        chosen_indices = np.random.choice(len(self.buffer),
                                          size=min(self.batch_size, len(self.buffer)),
                                          replace=False,
                                          p=p_indices)

        buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]

        return buffer


# replay buffer
class SumTree:
    # little modified from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    def __init__(self, capacity):
        self.capacity = capacity    # N, the size of replay buffer, so as to the number of sum tree's leaves
        self.tree = np.zeros(2 * capacity - 1)  # equation, to calculate the number of nodes in a sum tree
        self.transitions = np.empty(capacity, dtype=object)
        self.next_idx = 0

    @property
    def total_p(self):
        return self.tree[0]

    def add(self, priority, transition):
        idx = self.next_idx + self.capacity - 1
        self.transitions[self.next_idx] = transition
        self.update(idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)    # O(logn)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, s):
        idx = self._retrieve(0, s)   # from root
        trans_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.transitions[trans_idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])


# batch_size=8, target_update_iter=400, train_nums=5000, buffer_size=200, replay_period=20,
# alpha=0.4, beta=0.4, beta_increment_per_sample=0.001

# replay buffer params [(s, a, r, ns, done), ...]



# self.b_obs = np.empty((self.batch_size,) + self.env.reset().shape)
# self.b_actions = np.empty(self.batch_size, dtype=np.int8)
# self.b_rewards = np.empty(self.batch_size, dtype=np.float32)
# self.b_next_states = np.empty((self.batch_size,) + self.env.reset().shape)
# self.b_dones = np.empty(self.batch_size, dtype=np.bool)

# self.replay_buffer = SumTree(buffer_size)   # sum-tree data structure
# self.buffer_size = buffer_size              # replay buffer size N
# self.replay_period = replay_period          # replay period K
# self.alpha = alpha                          # priority parameter, alpha=[0, 0.4, 0.5, 0.6, 0.7, 0.8]
# self.beta = beta                            # importance sampling parameter, beta=[0, 0.4, 0.5, 0.6, 1]
# self.beta_increment_per_sample = beta_increment_per_sample
# self.num_in_buffer = 0                      # total number of transitions stored in buffer
# self.margin = 0.01                          # pi = |td_error| + margin
# self.p1 = 1                                 # initialize priority for the first transition
# # self.is_weight = np.empty((None, 1))
# self.is_weight = np.power(self.buffer_size, -self.beta)  # because p1 == 1
# self.abs_error_upper = 1