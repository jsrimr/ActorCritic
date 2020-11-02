
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


ENV_NAME = 'Pendulum-v0'
# Breakout-v0 대신 BreakoutNoFrameskip-v4을 사용
# v0은 2~4개 프레임을 자동으로 생략하므로 이 기능이 없는 버전을 사용한다
# 참고 URL https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L371

NUM_SKIP_FRAME = 4  # 생략할 프레임 수
NUM_STACK_FRAME = 4  # 하나의 상태로 사용할 프레임의 수
NOOP_MAX = 30  # 초기화 후 No-operation을 적용할 최초 프레임 수의 최댓값
NUM_PROCESSES = 16  # 병렬로 실행할 프로세스 수
NUM_ADVANCED_STEP = 5  # Advanced 학습할 단계 수
GAMMA = 0.99  # 시간할인율

TOTAL_FRAMES = 10e6  # 학습에 사용하는 총 프레임 수
NUM_UPDATES = int(TOTAL_FRAMES / NUM_ADVANCED_STEP /
                  NUM_PROCESSES)  # 신경망 수정 총 횟수
# NUM_UPDATES는 약 125,000이 됨


# A2C 손실함수를 계산하기 위한 상수
value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5

# 최적회 기법 RMSprop에 대한 설정
lr = 7e-4
eps = 1e-5
alpha = 0.99

# GPU 사용 설정
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# print(device)
