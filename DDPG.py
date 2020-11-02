import gym

from MLPmodel import ValueNetwork, PolicyNetwork, DDPG
from common.multiprocessing_env import SubprocVecEnv
from hyperparameters import *
from utils import NormalizedActions, OUNoise, ReplayBuffer

seed_num = 1
torch.manual_seed(seed_num)
if use_cuda:
    torch.cuda.manual_seed(seed_num)

# 실행환경 구축
torch.set_num_threads(seed_num)


def make_env(env_id):
    def _thunk():
        '''멀티 프로세스로 동작하는 환경 SubprocVecEnv를 실행하기 위해 필요하다'''
        env = gym.make(env_id)
        env = NormalizedActions(env)
        return env

    return _thunk


envs = [make_env(ENV_NAME) for i in range(NUM_PROCESSES)]
envs = SubprocVecEnv(envs)  # 멀티프로세스 실행환경

ou_noise = OUNoise(envs.action_space)

state_dim = envs.observation_space.shape[0]
action_dim = envs.action_space.shape[0]
hidden_dim = 256

value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

value_lr = 1e-3
policy_lr = 1e-4

# value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
# policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
ddpg = DDPG(value_net, policy_net, target_value_net, target_policy_net)
# value_criterion = nn.MSELoss()

replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)

max_frames = 12000 * NUM_PROCESSES
max_steps = 500
frame_idx = 0
episode_rewards = []
batch_size = 128

if __name__ == "__main__":
    # 초기 상태로 시작
    while frame_idx < max_frames:
        state = envs.reset()
        ou_noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = policy_net.get_action(state)
            action = ou_noise.get_action(action, step)
            next_state, reward, done, _ = envs.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                ddpg.update(batch_size, replay_buffer)

            state = next_state
            episode_reward += reward
            frame_idx += NUM_PROCESSES

            if frame_idx % (NUM_PROCESSES * 100) == 0:
                # plot(frame_idx, rewards)
                # rewards_tmp = np.array(rewards)
                if episode_rewards:
                    print("finished frames {}, {:.1f}".
                          format(frame_idx, episode_rewards[-1]))

            if done.any(): #todo : subproess 시에 도중에 끝나버리는 걸 어떻게 처리해야하지?
                break

        episode_rewards.append(episode_reward.mean())
