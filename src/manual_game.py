import pygame
from snake import Snake
pygame.init()

size = width, height = 600, 600
win = pygame.display.set_mode(size)
pygame.display.set_caption("Snake")
clock = pygame.time.Clock()

snake = Snake(20, width, height)

running = True
while running:
    if snake.update() == 1:
        pygame.quit()
        
    win.fill((51, 51, 51))
    
    snake.draw(win)

    pygame.display.update()
    
    key_pressed = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.KEYDOWN:
            key_pressed = event.key
            
            
    snake.direction(key_pressed)            

    clock.tick(15)