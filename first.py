from collections import deque
import gym
import os
import numpy as np
import torch
import random
import torch.functional as F

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=50000)
        self.batch_size = 32
        self.build_model()

    def build_model(self):
        inp = torch.nn.Linear(4, 12)
        h1 = torch.nn.Linear(12, 12)
        h2 = torch.nn.Linear(12, 12)
        out = torch.nn.Linear(12, 1)

        model = torch.nn.Sequential(inp, torch.nn.ReLU(),
                h1, torch.nn.ReLU(),
                h2, torch.nn.ReLU(),
                out)

        model.train()

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss(reduction='sum')


    def toggle_train(self):
        if self.model.training:
            self.model.eval()
        else:
            self.model.train()

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, do_train = True):
        if do_train:
            return random.randrange(self.action_size) # do a more intellegent sample here
        return np.argmax(self.model(state))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target_f = self.model(torch.from_numpy(state.astype(np.float32)))
            loss = self.criterion(target_f, torch.Tensor([[reward]]))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


env = gym.make("CartPole-v1")
agent = Agent(4, 2)
for o in range(500):
    observation = env.reset()
    final_reward = 0.0
    for t in range(100000):
        # env.render()
        state = observation.reshape(1, env.observation_space.shape[0])
        action = agent.select_action(state)
        observation, reward, done, info = env.step(action)
        final_reward += reward

        agent.add_to_memory(state, action, reward, observation, done)
        agent.train_step()

        if done:
            observation = env.reset()
    print("Final reward for episode {}: {}".format(o, final_reward))

output_file = os.environ.get("WEIGHTS_OUTPUT_FILE", "cartpolev1_torch_simple.pt")
torch.save(agent.model.state_dict(), output_file)

env.close()
