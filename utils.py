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