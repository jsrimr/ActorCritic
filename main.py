from utils import RolloutStorage
from hyperparameters import *
from common.multiprocessing_env import SubprocVecEnv
from model import Net, Brain
from envs import make_env
from tqdm import tqdm
import numpy as np

seed_num = 1
torch.manual_seed(seed_num)
if use_cuda:
    torch.cuda.manual_seed(seed_num)

# 실행환경 구축
torch.set_num_threads(seed_num)
envs = [make_env(ENV_NAME, seed_num, i) for i in range(NUM_PROCESSES)]
envs = SubprocVecEnv(envs)  # 멀티프로세스 실행환경

n_out = envs.action_space.n  # 행동의 가짓수는 4
actor_critic = Net(n_out).to(device)  # GPU 사용
global_brain = Brain(actor_critic)

# 정보 저장용 변수 생성
obs_shape = envs.observation_space.shape  # (1, 84, 84)
obs_shape = (obs_shape[0] * NUM_STACK_FRAME,
                *obs_shape[1:])  # (4, 84, 84)
# torch.Size([16, 4, 84, 84])
current_obs = torch.zeros(NUM_PROCESSES, *obs_shape).to(device)
rollouts = RolloutStorage(
    NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)  # rollouts 객체
episode_rewards = torch.zeros([NUM_PROCESSES, 1])  # 현재 에피소드에서 받을 보상 저장
final_rewards = torch.zeros([NUM_PROCESSES, 1])  # 마지막 에피소드의 총 보상 저장

if __name__ == "__main__":
    # 초기 상태로 시작  
    obs = envs.reset()
    obs = torch.from_numpy(obs).float()  # torch.Size([16, 1, 84, 84])
    current_obs[:, -1:] = obs  # 4번째 프레임에 가장 최근 관측결과를 저장

    # advanced 학습에 사용할 객체 rollouts에 첫번째 상태로 현재 상태를 저장
    rollouts.observations[0].copy_(current_obs)

     # 주 반복문
    for j in tqdm(range(NUM_UPDATES)):
        # advanced 학습 범위에 들어가는 단계마다 반복
        for step in range(NUM_ADVANCED_STEP):

            # 행동을 결정
            with torch.no_grad():
                action = actor_critic.act(rollouts.observations[step])

            cpu_actions = action.squeeze(1).cpu().numpy()  # tensor를 NumPy 변수로

            # 1단계를 병렬로 실행, 반환값 obs의 크기는 (16, 1, 84, 84)
            obs, reward, done, info = envs.step(cpu_actions)

            # 보상을 텐서로 변환한 다음 에피소드 총 보상에 더함
            # 크기가 (16,)인 것을 (16, 1)로 변환
            reward = np.expand_dims(np.stack(reward), 1)
            reward = torch.from_numpy(reward).float()
            episode_rewards += reward

            # 각 프로세스마다 done이 True이면 0, False이면 1
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            # 마지막 에피소드의 총 보상을 업데이트
            final_rewards *= masks  # done이 True이면 0을 곱하고, False이면 1을 곱하여 리셋
            # done이 False이면 0을 더하고, True이면 epicodic_rewards를 더함
            final_rewards += (1 - masks) * episode_rewards

            # 에피소드의 총 보상을 업데이트
            episode_rewards *= masks  # 각 프로세스마다 done이 True이면 0, False이면 1을 곱함

            # masks 변수를 GPU로 전달
            masks = masks.to(device)

            # done이 True이면 모두 0으로
            # mask의 크기를 torch.Size([16, 1]) --> torch.Size([16, 1, 1 ,1])로 변환하고 곱함
            current_obs *= masks.unsqueeze(2).unsqueeze(2)

            # 프레임을 모음
            # torch.Size([16, 1, 84, 84])
            obs = torch.from_numpy(obs).float()
            current_obs[:, :-1] = current_obs[:, 1:]  # 0～2번째 프레임을 1~3번째 프레임으로 덮어씀
            current_obs[:, -1:] = obs  # 4번째 프레임에 가장 최근 obs를 저장

            # 메모리 객체에 현 단계의 transition을 저장
            rollouts.insert(current_obs, action.data, reward, masks)
# advanced 학습의 for문 끝

        # advanced 학습 대상 단계 중 마지막 단계의 상태에서 예상되는 상태가치를 계산
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.observations[-1]).detach()

        # 모든 단계의 할인 총보상을 계산하고, rollouts의 변수 returns를 업데이트
        rollouts.compute_returns(next_value)

        # 신경망 수정 및 rollout 업데이트
        global_brain.update(rollouts)
        rollouts.after_update()

        # 로그 기록 : 중간 결과 출력
        if j % 100 == 0:
            print("finished frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                    format(j*NUM_PROCESSES*NUM_ADVANCED_STEP,
                            final_rewards.mean(),
                            final_rewards.median(),
                            final_rewards.min(),
                            final_rewards.max()))

        # 결합 가중치 저장
        if j % 12500 == 0:
            torch.save(global_brain.actor_critic.state_dict(),
                        'weight_'+str(j)+'.pth')

        # 주 반복문 끝
        torch.save(global_brain.actor_critic.state_dict(), 'weight_end.pth')