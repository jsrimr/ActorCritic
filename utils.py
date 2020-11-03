import random

import numpy as np

from hyperparameters import *


class RolloutStorage(object):
    '''Advantage 학습에 사용하는 메모리 클래스'''

    def __init__(self, num_steps, num_processes, obs_shape):
        self.observations = torch.zeros(
            num_steps + 1, num_processes, *obs_shape).to(device)
        # *로 리스트의 요소를 풀어낸다(unpack)
        # obs_shape→(4,84,84)
        # *obs_shape→ 4 84 84

        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(
            num_steps, num_processes, 1).long().to(device)

        # 할인 총보상을 저장
        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.index = 0  # 저장할 인덱스

    def insert(self, current_obs, action, reward, mask):
        '''인덱스가 가리키는 다음 자리에 transition을 저장'''
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # 인덱스 업데이트

    def after_update(self):
        '''Advantage 학습 단계 수만큼 단계가 진행되면 가장 최근 단계를 index0에 저장'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''Advantage 학습 단계에 들어가는 각 단계에 대해 할인 총보상을 계산'''

        # 주의 : 5번째 단계부터 거슬러 올라가며 계산
        # 주의 : 5번째 단계가 Advantage1, 4번째 단계가 Advantage2가 되는 식임
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * \
                                    GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


import gym


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py

def compute_gae(next_value, rewards, masks, values, gamma=0.99, lmbda=0.95):
    # values = torch.cat([values, next_value], dim=-1)

    returns = []
    td_target = rewards + gamma * next_value * masks
    delta = td_target - values
    advantage = 0.0
    for delta_t in reversed(delta):
        advantage = gamma * lmbda * advantage + delta_t
        returns.append([advantage.mean()])

    returns.reverse()
    return torch.tensor(returns, requires_grad = True)
