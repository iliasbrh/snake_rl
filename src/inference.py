import pygame
import torch
import model
pygame.init()

size = width, height = 600, 600
win = pygame.display.set_mode(size)
pygame.display.set_caption("Snake")
clock = pygame.time.Clock()

env = model.Environment(width, height)
policy = model.DQN()
policy.load_state_dict(torch.load("models/model2.pth", map_location=torch.device("cpu"), weights_only=False))

s, _, terminated, _ = env.reset()

def select_action_determinist(state):
    with torch.no_grad():
        return torch.argmax(policy(state)).item()

running = True
while running:
    if terminated:
        s, _, terminated, _ = env.reset()
    else:
        action = select_action_determinist(s)
        s, _, terminated, _ = env.step(action)
        
    win.fill((51, 51, 51))
    
    env.snake_game.draw(win)

    pygame.display.update()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False        

    clock.tick(120)


pygame.quit() 