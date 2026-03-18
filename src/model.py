import pygame
import numpy as np
from statistics import mean
from snake import Snake
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import random
from collections import deque

window_size = width, height = 200, 200
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
            "state": torch.stack([transition[0] for transition in batch]),
            "action": torch.tensor([transition[1] for transition in batch], dtype=torch.int64, device=device).unsqueeze(1),
            "next_state": torch.stack([transition[2] for transition in batch]),
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
            reward = -40
            terminated = True
        elif outcome == 2:
            reward = 40
        else:
            reward = -0.05
        
        state = torch.zeros((10), dtype=torch.float32, device=device)
        state[0] = (self.snake_game.apple.position[0] - self.snake_game.positions[-1][0])/self.width
        state[1] = (self.snake_game.apple.position[1] - self.snake_game.positions[-1][1])/self.height
        
        for i, (dx, dy) in enumerate(directions):
            next_x = self.snake_game.positions[-1][0] + dx * snake_size
            next_y = self.snake_game.positions[-1][1] + dy * snake_size
            
            wall = next_x >= self.width or next_x < 0 or next_y >= self.height or next_y < 0
            body = (next_x, next_y) in set(self.snake_game.positions[:-1])
            state[i+2] = float(wall or body)
            
            state[i+6] = int((dx, dy) == (self.snake_game.speedx, self.snake_game.speedy))
        
        """
        state = torch.zeros((n_observations, width//snake_size, height//snake_size), dtype=torch.float32, device=device)
        state[0, (self.snake_game.apple.position[1]//snake_size)%10, (self.snake_game.apple.position[0]//snake_size)%10] = 1.
        state[1, (self.snake_game.positions[-1][1]//snake_size)%10, (self.snake_game.positions[-1][0]//snake_size)%10] = 1.
        for pos in self.snake_game.positions[:-1]:
            state[2, pos[1]//snake_size, pos[0]//snake_size] = 1.
        """
        return state, reward, terminated, len(self.snake_game.positions)
    
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

n_observations = 10
n_actions = 4

class DQN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        self.conv1 = torch.nn.Conv2d(n_observations, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1) 
        """
        self.linear1 = torch.nn.Linear(n_observations, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, n_actions)
        
        self.flatten = torch.nn.Flatten()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.view(-1, ((width//snake_size)//4)*((height//snake_size)//4)*32)
        """
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
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50000
TAU = 0.005
LR = 1e-4




if __name__ == "__main__":
    num_episode = 50000
    warmup_episode = 7000

    memory = ReplayMemory(100000)
    step = 0
    env = Environment()

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    """
    step between episodes instead
    scheduler_warmup = LinearLR(optimizer, start_factor=0.33, end_factor=1.0, total_iters=warmup_episode)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_episode - warmup_episode, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_episode])
    """
    
    
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
        
        # scheduler.step()
        length_history.append(length)
        
        if i%10 == 0:
            print("="*50, flush=True)
            print(f"Episode {i+1}", flush=True)
            print(f"Eps = {EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY)}", flush=True)
            # print(f"Learning rate = {scheduler.get_last_lr()}", flush=True)
            print(f"Average score on recent episodes : {mean(length_history[-100:])}", flush=True)
            
    torch.save(policy_net.state_dict(), "models/model2.pth")