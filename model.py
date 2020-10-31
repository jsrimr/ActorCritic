import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperparameters import *


def init(module, gain):
    '''결합 가중치를 초기화하는 함수'''
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):
    '''합성곱층의 출력 이미지를 1차원으로 변환하는 층'''

    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, n_out):
        super(Net, self).__init__()

        # 결합 가중치 초기화 함수
        def init_(module): return init(
            module, gain=nn.init.calculate_gain('relu'))

        # 합성곱층을 정의
        self.conv = nn.Sequential(
            # 이미지 크기의 변화 (84*84 -> 20*20)
            init_(nn.Conv2d(NUM_STACK_FRAME, 32, kernel_size=8, stride=4)),
            # 프레임 4개를 합치므로 input=NUM_STACK_FRAME=4가 된다. 출력은 32이다.
            # size 계산  size = (Input_size - Kernel_size + 2*Padding_size)/ Stride_size + 1

            nn.ReLU(),
            # 이미지 크기의 변화 (20*20 -> 9*9)
            init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            # 이미지 크기의 변화(9*9 -> 7*7)
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            Flatten(),  # 이미지를 1차원으로 변환
            init_(nn.Linear(64 * 7 * 7, 512)),  # 7*7 이미지 64개를 512차원으로 변환
            nn.ReLU()
        )

        # 결합 가중치 초기화 함수
        def init_(module): return init(module, gain=1.0)

        # Critic을 정의
        self.critic = init_(nn.Linear(512, 1))  # 출력은 상태가치이므로 1개

        # 결합 가중치 초기화 함수
        def init_(module): return init(module, gain=0.01)

        # Actor를 정의
        self.actor = init_(nn.Linear(512, n_out))  # 출력이 행동이므로 출력 수는 행동의 가짓수

        # 신경망을 학습 모드로 전환
        self.train()

    def forward(self, x):
        '''신경망의 순전파 계산 정의'''
        input = x / 255.0  # 이미지의 픽셀값을 [0,255]에서 [0,1] 구간으로 정규화
        conv_output = self.conv(input)  # 합성곱층 계산
        critic_output = self.critic(conv_output)  # 상태가치 출력 계산
        actor_output = self.actor(conv_output)  # 행동 출력 계산

        return critic_output, actor_output

    def act(self, x):
        '''상태 x일때 취할 확률을 확률적으로 구함'''
        value, actor_output = self(x)
        probs = F.softmax(actor_output, dim=1)    # dim=1で行動の種類方向に計算
        action = probs.multinomial(num_samples=1)

        return action

    def get_value(self, x):
        '''상태 x의 상태가치를 구함'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''상태 x의 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 구함'''
        value, actor_output = self(x)

        # dim=1이므로 행동의 종류 방향으로 계산
        log_probs = F.log_softmax(actor_output, dim=1)
        action_log_probs = log_probs.gather(
            1, actions)  # 실제 행동에 대한 log_probs 계산

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류 방향으로 계산
        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, dist_entropy


# 에이전트의 두뇌 역할을 하는 클래스로, 모든 에이전트가 공유한다


class Brain(object):
    def __init__(self, actor_critic):

        self.actor_critic = actor_critic  # actor_critic은 Net클래스로 구현한 신경망이다

        # 이미 학습된 결합 가중치를 로드하려면
        # filename = 'weight.pth'
        # param = torch.load(filename, map_location='cpu')
        # self.actor_critic.load_state_dict(param)

        # 가중치를 학습하는 최적화 알고리즘 설정
        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr=lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        '''advanced 학습 대상 5단계를 모두 사용하여 수정한다'''
        obs_shape = rollouts.observations.size()[2:]  # torch.Size([4, 84, 84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1))

        # 각 변수의 크기에 주의할 것
        # rollouts.observations[:-1].view(-1, *obs_shape) torch.Size([80, 4, 84, 84])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # dist_entropy torch.Size([])

        values = values.view(num_steps, num_processes,
                             1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values  # torch.Size([5, 16, 1])
        value_loss = advantages.pow(2).mean()

        action_gain = (advantages.detach() * action_log_probs).mean()
        # advantages는 detach 하여 정수로 취급한다

        total_loss = (value_loss * value_loss_coef -
                      action_gain - dist_entropy * entropy_coef)

        self.optimizer.zero_grad()  # 경사 초기화
        total_loss.backward()  # 역전파 계산
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        # 한번에 결합 가중치가 너무 크게 변화하지 않도록, 경사의 최댓값을 0.5로 제한한다

        self.optimizer.step()  # 결합 가중치 수정
