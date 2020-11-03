import torch
import gym
from MLPmodel import PolicyNetwork
# from utils import NormalizedActions
from utils import NormalizedActions

if __name__ == '__main__':
    env = NormalizedActions(gym.make("Pendulum-v0"))
    state = env.reset()

    model = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0], 256)
    # model.load_state_dict(torch.load("DDPG_pendulum_weight.pth"))
    model.load_state_dict(torch.load("DDPG_original_pendulum_weight.pth"))

    episode_reward = 0
    while True:
        with torch.no_grad():
            action = model.get_action(state)
        state, reward, _, _ = env.step(action)

        episode_reward += reward
        print(episode_reward)

        env.render()