import pygame
from random import choice

class Apple:
    def __init__(self, size, width, height):
        self.allowed_pos = set([(x, y) for x in range(0, width, size) for y in range(0, height, size)])
        self.allowed_pos.remove(((width//2)//size*size, (height//2)//size*size))
        
        self.position = choice(list(self.allowed_pos))
        
    def move(self):
        self.position = choice(list(self.allowed_pos))
        
class Snake:
    def __init__(self, size, width, height):
        self.size = size
        
        self.positions = [((width//2)//size*size, (height//2)//size*size)]
        
        self.speedx = 1
        self.speedy = 0
        
        self.apple = Apple(size, width, height)
        
        self.width = width
        self.height = height
        
    def update(self):
        new_pos = (self.positions[-1][0] + self.size*self.speedx, self.positions[-1][1] + self.size*self.speedy)
        
        if new_pos in self.positions[:-1] or new_pos[0] >= self.width or new_pos[0] < 0 or new_pos[1] >= self.height or new_pos[1] < 0:
            return 1 # end of the game
        
        self.positions.append(new_pos)
        self.apple.allowed_pos.remove(new_pos)
        
        if new_pos == self.apple.position:
            self.apple.move()
            return 2 # to detect apple reward
        else:
            tail = self.positions.pop(0)
            self.apple.allowed_pos.add(tail)
            return 0
            
    def direction(self, key):
        if key == pygame.K_UP:
            if self.speedy != 1: # the y axis is towards the bottom
                self.speedy = -1
                self.speedx = 0
        elif key == pygame.K_DOWN:
            if self.speedy != -1:
                self.speedy = 1
                self.speedx = 0
        elif key == pygame.K_RIGHT:
            if self.speedx != -1:
                self.speedy = 0
                self.speedx = 1
        elif key == pygame.K_LEFT:
            if self.speedx != 1:
                self.speedy = 0
                self.speedx = -1
        
    def draw(self, win):
        for pos in self.positions[:-1]:
            pygame.draw.rect(win, (30, 240, 10), (pos[0], pos[1], self.size, self.size))
        pygame.draw.rect(win, (255, 255, 255), (self.positions[-1][0], self.positions[-1][1], self.size, self.size))
        pygame.draw.rect(win, (230, 20, 10), (self.apple.position[0], self.apple.position[1], self.size, self.size))
        
    