import pygame
import numpy as np
from statistics import mean
from snake import Snake
import torch
import random
from collections import deque

snake_size = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush=True)


##########################
#   TRANSITION MEMORY    #
##########################

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return {
            "state": (torch.cat([transition[0][0] for transition in batch], dim=0), torch.cat([transition[0][1] for transition in batch], dim=0)), # for the state_map and state_direction
            "action": torch.tensor([transition[1] for transition in batch], dtype=torch.int64, device=device).unsqueeze(1),
            "next_state": (torch.cat([transition[2][0] for transition in batch], dim=0), torch.cat([transition[2][1] for transition in batch], dim=0)),
            "reward": torch.tensor([transition[3] for transition in batch], dtype=torch.float32, device=device).unsqueeze(1),
            "done": torch.tensor([transition[4] for transition in batch], dtype=torch.float32, device=device).unsqueeze(1)
        }

    def __len__(self):
        return len(self.memory)
    
####################
#   ENVIRONMENT    #
####################
actions_lookuptable = [pygame.K_UP, pygame.K_DOWN, pygame.K_RIGHT, pygame.K_LEFT]
directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]

class Environment:
    def __init__(self, width, height):
        self.snake_game = None
        self.width = width
        self.height = height
    
    def get_observation(self):
        outcome = self.snake_game.update()
        terminated = False
        
        # outcome 1 -> game lost
        # outcome 2 -> apple eaten
        # otherwise -> nothing special, and a negative reward to make sure the snake does not just 
        # wander around
        if outcome == 1:
            reward = -10
            terminated = True
        elif outcome == 2:
            reward = 10
        else:
            reward = -0.01
        
        state_map = torch.zeros((1, n_observations, width//snake_size, height//snake_size), dtype=torch.float32, device=device)
        state_map[0, 0, (self.snake_game.apple.position[1]//snake_size)%10, (self.snake_game.apple.position[0]//snake_size)%10] = 1.
        state_map[0, 1, (self.snake_game.positions[-1][1]//snake_size)%10, (self.snake_game.positions[-1][0]//snake_size)%10] = 1.
        for pos in self.snake_game.positions[:-1]:
            state_map[0, 2, pos[1]//snake_size, pos[0]//snake_size] = 1.

        state_direction = torch.zeros((1, 4), dtype=torch.float32, device=device)
        for i, (dx, dy) in enumerate(directions):
            state_direction[0, i] = int((dx, dy) == (self.snake_game.speedx, self.snake_game.speedy))

        return (state_map, state_direction), reward, terminated, len(self.snake_game.positions)
    
    def step(self, action):
        # action is in the format [0, 1, 2, 3] that we must convert to UP DOWN RIGHT LEFT
        signal = actions_lookuptable[action]
        self.snake_game.direction(signal)
        return self.get_observation()
        
    def reset(self):
        self.snake_game = Snake(20, self.width, self.height)
        return self.get_observation()
        
##############
#   MODEL    #
##############

# 2 model ideas : 
# - a CNN that takes in the data of each pixel of the map
# - a feed forward that takes in 10 observations : 
#       - the dx and dy from the snake's head to the apple
#       - 4 one hot signals for the current direction
#       - 4 signals saying if the adjacent square in the given direction is occupied or is the edge of the map

n_observations = 3
n_actions = 4

class DQN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(n_observations, 64, 5, padding='same')
        self.conv2 = torch.nn.Conv2d(64, 64, 5, padding='same') 
        
        self.linear1 = torch.nn.Linear(260, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, n_actions)
        
        self.flatten = torch.nn.Flatten()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x_map, x_direction = x
        
        x = self.conv1(x_map)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.view(-1, 256)
        x = torch.cat([x, x_direction], dim=1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        
        return x # predicted q-values for each of the 4 actions


def select_action(state):
    # implementing epsilon decay to make the initial versions discover things randomly
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY)
    random_value = random.random()
    if random_value <= eps:
        return random.choice([0, 1, 2, 3])
    else:
        with torch.no_grad():
            return torch.argmax(policy_net(state)).item()
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    state = transitions["state"]
    action = transitions["action"]
    next_state = transitions["next_state"]
    reward = transitions["reward"]
    done = transitions["done"]
    
    current_q_values = policy_net(state).gather(1, action)

    with torch.no_grad():
        next_q_values = target_net(next_state).max(1)[0].unsqueeze(1)

    target_q_values = reward + (GAMMA * next_q_values * (1 - done))
    loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    
def soft_update():
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)


###################
# HYPERPARAMETERS #
###################

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 100000
TAU = 0.005
LR = 4e-5


if __name__ == "__main__":
    window_size = width, height = 200, 200
    
    num_episode = 10000
    warmup_episode = 700

    memory = ReplayMemory(100000)
    step = 0
    env = Environment(width, height)

    checkpoint = torch.load("models/checkpoint1.pth")
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    
    # target_net.load_state_dict(policy_net.state_dict())
    policy_net.load_state_dict(checkpoint['policy'])
    target_net.load_state_dict(checkpoint['target'])

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    ######################
    #   TRAINING LOOP    #
    ###################### 

    length_history = []

    for i in range(num_episode):
        observation = env.reset()[0]
        done = False
        while not done:
            action = select_action(observation)
            next_observation, reward, terminated, length = env.step(action)
            done = int(terminated)

            memory.push(observation, action, next_observation, reward, done)
            optimize_model()

            soft_update()
            observation = next_observation

            step += 1
        
        length_history.append(length)
        
        if i%10 == 0:
            print("="*50, flush=True)
            print(f"Episode {i+1}", flush=True)
            print(f"Eps = {EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY)}", flush=True)
            print(f"Average score on recent episodes : {mean(length_history[-100:])}", flush=True)
    checkpoint = { 
        'policy': policy_net.state_dict(),
        'target': target_net.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'models/checkpoint2.pth')
