import numpy as np

from hyperparameters import *
from utils import compute_gae


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        # state = torch.FloatTensor(state).unsqueeze(0).to(device)
        state = torch.FloatTensor(state).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()


class DDPG:
    def __init__(self, value_net, policy_net, target_value_net, target_policy_net, gamma=0.99, soft_tau=1e-2, ):
        self.gamma = .99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 1e-2
        self.value_net = value_net
        self.policy_net = policy_net
        self.target_value_net = target_value_net
        self.target_policy_net = target_policy_net

        self.gamma = 0.99
        self.soft_tau = 1e-2
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=1e-4)
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=1e-3)

    def update(self, batch_size, replay_buffer):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(device)
        # reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
        # done = torch.FloatTensor(np.float32(done)).to(device)

        # policy_loss = self.value_net(state, self.policy_net(state))
        values = self.value_net(state, self.policy_net(state))

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        #
        policy_loss = compute_gae(target_value, reward, done, values)
        policy_loss = -policy_loss.mean()
        # expected_value = torch.clamp(expected_value, min_value, max_value)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        value = self.value_net(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())


        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )
