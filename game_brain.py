import random
from collections import deque
import torch
from torch import nn
class GameBrain(nn.Module):
    def __init__(self, state_size=4, action_size=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, state):
        return self.network(state)

class DQNAgent:
    def __init__(self, state_size=4, action_size=3):
        self.brain = GameBrain(state_size, action_size)
        self.target_brain = GameBrain(state_size, action_size)
        self.target_brain.load_state_dict(self.brain.state_dict())
        self.target_brain.eval()

        self.optimizer = torch.optim.Adam(
            self.brain.parameters(), lr=0.001
        )
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=2000)
        self.action_size = action_size

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.train_steps = 0
        self.target_update_interval = 100

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32
            )
            values = self.brain(state_tensor)

        return int(torch.argmax(values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)

        states = torch.tensor(
            [x[0] for x in batch], dtype=torch.float32
        )
        actions = torch.tensor(
            [x[1] for x in batch], dtype=torch.long
        )
        rewards = torch.tensor(
            [x[2] for x in batch], dtype=torch.float32
        )
        next_states = torch.tensor(
            [x[3] for x in batch], dtype=torch.float32
        )
        dones = torch.tensor(
            [x[4] for x in batch], dtype=torch.float32
        )

        current = self.brain(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            future = self.target_brain(
                next_states
            ).max(dim=1).values

            target = rewards + self.gamma * future * (1 - dones)

        loss = self.loss_fn(current, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update_interval == 0:
            self.target_brain.load_state_dict(
                self.brain.state_dict()
            )

            self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

        return loss.item()

if __name__ == "__main__":
    agent = DQNAgent()
    state = [0.0, 0.0, 1.0, 0.0]
    action = agent.choose_action(state)

    print("Состояние игры:", state)
    print("Выбранное действие:", action)