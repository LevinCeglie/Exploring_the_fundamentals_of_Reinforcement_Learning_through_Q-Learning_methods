##############################################################
# This code contains my self-made Snake environment
# adjusted to be used as a Markov decision process
##############################################################


import random
import numpy as np
import torch
import drawSvg as draw

class Snake:
    def __init__(self, spawnx, spawny):
        #only considers the head and its movement
        self.x = spawnx
        self.y = spawny
        self.changelist = []


    def act(self, action):
        #takes action and keep a record of it in the changelist
        # for delayed body movement
        if action == 0:
            self.x += 1
            self.changelist.append((1, 0))
        if action == 1:
            self.x -= 1
            self.changelist.append((-1, 0))
        if action == 2:
            self.y += 1
            self.changelist.append((0, 1))
        if action == 3:
            self.y -= 1
            self.changelist.append((0, -1))

class Body:
    def __init__(self, len):
        #keeps track of the snake's body
        self.len = len
        self.list = []

    def grow(self, snake):
        #function to grow the body
        self.len += 1
        self.list.append([snake.x,snake.y])
    
    def update(self, changelist, snake):
        #function to position each body part at its position
        #uses the snake's head position and the changelist 
        # to calculate the position
        for i in range(0,self.len):
            x = snake.x
            y = snake.y
            for n in range(1,(i+2)):
                ch = changelist[-n]
                x = x-ch[0]
                y = y-ch[1]     
            self.list[i] = ([x,y])

class Fruit:
    #used for the fruit
    def __init__(self, gridlength, spawnset = False, 
                            spawnx = None , spawny = None):
        #spawns the fruit at a either given or random position 
        if spawnset:
            self.x = spawnx
            self.y = spawny
        else:
            self.x = random.randint(0,(gridlength-1))
            self.y = random.randint(0,(gridlength-1))

class Environment:
    def __init__(self, gridlength, snakex, snakey, r_fruit, 
                r_collision, r_step, fruitspawnset,  
                fruitx = None, fruity = None, statetype = 1):
        
        #initiliazing variables, snake, fruit and body
        self.gridlength = gridlength
        self.snakex = snakex
        self.snakey = snakey
        self.snakecolor = '#197416'
        self.fruitcolor = '#E92323'
        self.bodycolor = '#32B145'
        self.background = '#485B57'
        self.fruitx = fruitx
        self.fruity = fruity
        self.fruitspawnset = fruitspawnset
        self.snake = Snake(self.snakex, self.snakey)
        self.fruit = Fruit(self.gridlength, 
                            spawnset=self.fruitspawnset, 
                            spawnx=self.fruitx, spawny=self.fruity)
        self.body = Body(0)
        self.score = 0
        self.scorelist = []
        self.gamewon = False
        self.statetype = statetype

        #adjustable rewards given to the agent
        self.r_fruit = r_fruit
        self.r_collision = r_collision
        self.r_step = r_step

    def collisionfruit(self):
        #checks whether the snake has collided with the fruit
        if self.snake.x == self.fruit.x and self.snake.y == self.fruit.y:
            return True

    def collisionborder(self):
        #checks whether the snake has collided with the border
        if (self.snake.x == self.gridlength or self.snake.x == -1 or 
            self.snake.y == self.gridlength or self.snake.y == -1):
            return True
    
    def collisionself(self):
        #checks whether the snake has collided with its body
        for b in self.body.list:
            if self.snake.x == b[0] and self.snake.y == b[1]:
                return True

    def check_fruit_spawn(self):
        #makes sure the fruit can only spawn on a free cell
        self.fruit = Fruit(self.gridlength)
        for b in self.body.list:
            if self.fruit.x == b[0] and self.fruit.y == b[1]:
                self.check_fruit_spawn()
            else:
                pass
        if self.fruit.x == self.snake.x and self.fruit.y == self.snake.y:
            self.check_fruit_spawn()
        else:
            pass

    def get_state(self):
        #this function is needed, 
        # because there are different state types to choose from
        if self.statetype == 1:
            return self.get_state1()
        elif self.statetype == 2:
            return self.get_state2()
        elif self.statetype == 3:
            return self.get_state3()
        elif self.statetype == 4:
            return self.get_state4()
        elif self.statetype == 5:
            return self.get_state5()
        elif self.statetype == 6:
            return self.get_state6()
        elif self.statetype == 7:
            return self.get_state7()
        elif self.statetype == 8:
            return self.get_state8()

    def get_state1(self):
        #returns a 24-dimensional vector,
        #consisting of the distances between the snake's head and:
        #the border
        #+ binary vision for the fruit and its body
        #for each object 8 different directions get checked:
        #if there is no object in that direction the distance is 0
        #border distances to walls
        #and diagonals

        #walls, if it is 1 then one more step in that direction is lethal
        b0 = float(self.gridlength-self.snake.x)
        b2 = float(self.snake.y+1)
        b4 = float(self.snake.x+1)
        b6 = float(self.gridlength-self.snake.y)
        
        b1 = False
        b3 = False
        b5 = False
        b7 = False
        #diagonals
        for i in range(self.gridlength):
            if ((self.snake.x+i >= self.gridlength or 
                self.snake.y+i >= self.gridlength) and 
                (not b7)):
                b7 = float(np.sqrt(i**2+i**2))
            if ((self.snake.x-i < 0 or 
                self.snake.y-i < 0) and 
                (not b3)):
                b3 = float(np.sqrt(i**2+i**2))
            if ((self.snake.x+i >= self.gridlength or 
                self.snake.y-i < 0) and 
                (not b1)):
                b1 = float(np.sqrt(i**2+i**2))
            if ((self.snake.x-i < 0 or 
                self.snake.y+i >= self.gridlength) and 
                (not b5)):
                b5 = float(np.sqrt(i**2+i**2))

        f0 = 0.0
        f1 = 0.0
        f2 = 0.0
        f3 = 0.0
        f4 = 0.0
        f5 = 0.0
        f6 = 0.0
        f7 = 0.0

        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        s7 = 0.0

        if self.snake.y == self.fruit.y:
            if self.snake.x < self.fruit.x:
                f0 = 1.0
            else:
                f4 = 1.0
        if self.snake.x == self.fruit.x:
            if self.snake.y < self.fruit.y:
                f6 = 1.0
            else:
                f2 = 1.0


        for i in range(self.gridlength):
            if (self.snake.x+i == self.fruit.x and 
                self.snake.y+i == self.fruit.y):
                f7 = 1.0
            if (self.snake.x-i == self.fruit.x and 
                self.snake.y-i == self.fruit.y):
                f3 = 1.0
            if (self.snake.x+i == self.fruit.x and 
                self.snake.y-i == self.fruit.y):
                f1 = 1.0
            if (self.snake.x-i == self.fruit.x and 
                self.snake.y+i == self.fruit.y):
                f5 = 1.0
        
        for b in self.body.list:
            if self.snake.y == b[1]:
                if self.snake.x < b[0]:
                    s0 = 1.0
                else:
                    s4 = 1.0
            if self.snake.x == b[0]:
                if self.snake.y < b[1]:
                    s6 = 1.0
                else:
                    s2 = 1.0
            for i in range(self.gridlength):
                if (self.snake.x+i == b[0] and 
                    self.snake.y+i == b[1]):
                    s7 = 1.0
                if (self.snake.x-i == b[0] and 
                    self.snake.y-i == b[1]):
                    s3 = 1.0
                if (self.snake.x+i == b[0] and 
                    self.snake.y-i == b[1]):
                    s1 = 1.0
                if (self.snake.x-i == b[0] and 
                    self.snake.y+i == b[1]):
                    s5 = 1.0

        return torch.tensor([b0, b1, b2, b3, b4, b5, b6, b7,
                            f0, f1, f2, f3, f4, f5, f6, f7, 
                            s0, s1, s2, s3, s4, s5, s6, s7])
    
    def get_state2(self):
        #four directions for distances to border
        #four directions for binary vision to body
        #relative coordinates to fruit
        #up, down, right and left

        #border
        b0 = float(self.gridlength-self.snake.x)
        b1 = float(self.snake.y+1)
        b2 = float(self.snake.x+1)
        b3 = float(self.gridlength-self.snake.y)

        #own body
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0

        for b in self.body.list:
            if self.snake.x == b[0]:
                if self.snake.y >= b[1]:
                    s1 = 1.0
                if self.snake.y <= b[1]:
                    s3 = 1.0
            if self.snake.y == b[1]:
                if self.snake.x <= b[0]:
                    s0 = 1.0
                if self.snake.x >= b[0]:
                    s2 = 1.0

        #fruit
        fx = float(self.fruit.x - self.snake.x)
        fy = float(self.fruit.y - self.snake.y)
        return torch.tensor([b0, b1, b2, b3,
                            s0, s1, s2, s3,
                            fx, fy])

    def get_state3(self):
        #for border and body  in 4 directions
        #border with distances and body with binary vision
        #for the fruit a binary vision in 8 directions

        #border
        b0 = float(self.gridlength-self.snake.x)
        b1 = float(self.snake.y+1)
        b2 = float(self.snake.x+1)
        b3 = float(self.gridlength-self.snake.y)
          
        #body
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0

        for b in self.body.list:
            if self.snake.x == b[0]:
                if self.snake.y >= b[1]:
                    s1 = 1.0
                if self.snake.y <= b[1]:
                    s3 = 1.0
            if self.snake.y == b[1]:
                if self.snake.x <= b[0]:
                    s0 = 1.0
                if self.snake.x >= b[0]:
                    s2 = 1.0

        #fruit
        f0 = 0.0
        f1 = 0.0
        f2 = 0.0
        f3 = 0.0
        f4 = 0.0
        f5 = 0.0
        f6 = 0.0
        f7 = 0.0

        if self.snake.x == self.fruit.x:
            if self.snake.y >= self.fruit.y:
                f2 = 1.0
            if self.snake.y <= self.fruit.y:
                f6 = 1.0
        if self.snake.y == self.fruit.y:
            if self.snake.x <= self.fruit.x:
                f0 = 1.0
            if self.snake.x >= self.fruit.x:
                f4 = 1.0
        for i in range(-self.gridlength, self.gridlength):
            if ((self.snake.x+i) == self.fruit.x and 
                        (self.snake.y+i) == self.fruit.y):
                if i >= 0:
                    f7 = 1.0
                if i <= 0:
                    f3 = 1.0
            if ((self.snake.x+i) == self.fruit.x and 
                        (self.snake.y-i) == self.fruit.y):
                if i >= 0:
                    f1 = 1.0
                if i <= 0:
                    f5 = 1.0

        return torch.tensor([b0, b1, b2, b3, 
                            s0, s1, s2, s3, 
                            f0, f1, f2, f3, f4, f5, f6, f7])

    def get_state4(self):
        #capture the whole grid as 2d vector
        #snake = -0.5
        #fruit = 1
        #body = -1
        state = np.zeros((self.gridlength, self.gridlength))

        #to omit any erros as the snake-head tried to leave the grid
        if not self.snake.x >= self.gridlength or self.snake.x < 0:
            if not self.snake.y >= self.gridlength or self.snake.y < 0:
                state[self.snake.y][self.snake.x] = -0.5
        state[self.fruit.y][self.fruit.x] = 1
        for b in self.body.list:
            state[b[1]][b[0]] = -1
        
        state = torch.tensor(state)
        state = torch.flatten(state)
        state = state.float()
        return state  

    def get_state5(self):
        #captures a 5by5 2d array around the snakes head,
        #body = -1
        #border = -1
        #fruit = 1
        state = np.zeros((5,5))
        for i in range(-2,3):
            for j in range(-2,3):
                x = self.snake.x + i
                y = self.snake.y + j

                if (x >= self.gridlength or x < 0 or
                    y >= self.gridlength or y < 0):
                    state[j+2][i+2] = -1
                for b in self.body.list:
                    if x == b[0] and y == b[1]:
                        state[j+2][i+2] = -1
                if x == self.fruit.x and y == self.fruit.y:
                    state[j+2][i+2] = 1
        
        state = torch.tensor(state).float()
        return torch.flatten(state)
    
    def get_state6(self):
        #same as get_state5 
        #+ the vector from the snakes head to the fruit
        state = np.zeros((5,5))
        for i in range(-2,3):
            for j in range(-2,3):
                x = self.snake.x + i
                y = self.snake.y + j

                if (x >= self.gridlength or x < 0 or
                    y >= self.gridlength or y < 0):
                    state[j+2][i+2] = -1
                for b in self.body.list:
                    if x == b[0] and y == b[1]:
                        state[j+2][i+2] = -1
                if x == self.fruit.x and y == self.fruit.y:
                    state[j+2][i+2] = 1
        
        state = torch.tensor(state).float()
        state = torch.flatten(state)
        
        head_to_fruit = torch.tensor([self.fruit.x-self.snake.x, 
                                    self.fruit.y-self.snake.y]).float()

        return torch.cat((state, head_to_fruit))
    
    def get_state7(self):
        #captures a 9by9 2d array around the snakes head,
        #body = -1
        #border = -1
        #fruit = 1
        state = np.zeros((9,9))
        for i in range(-4,5):
            for j in range(-4,5):
                x = self.snake.x + i
                y = self.snake.y + j

                if (x >= self.gridlength or x < 0 or
                    y >= self.gridlength or y < 0):
                    state[j+4][i+4] = -1
                for b in self.body.list:
                    if x == b[0] and y == b[1]:
                        state[j+4][i+4] = -1
                if x == self.fruit.x and y == self.fruit.y:
                    state[j+4][i+4] = 1
        
        state = torch.tensor(state).float()
        state = torch.flatten(state)
        
        head_to_fruit = torch.tensor([self.fruit.x-self.snake.x, 
                                self.fruit.y-self.snake.y]).float()

        return torch.cat((state, head_to_fruit))

    def get_state8(self):
        #same as get_state7 
        #+ the vector from the snakes head to the fruit
        state = np.zeros((9,9))
        for i in range(-4,5):
            for j in range(-4,5):
                x = self.snake.x + i
                y = self.snake.y + j

                if (x >= self.gridlength or x < 0 or
                    y >= self.gridlength or y < 0):
                    state[j+4][i+4] = -1
                for b in self.body.list:
                    if x == b[0] and y == b[1]:
                        state[j+4][i+4] = -1
                if x == self.fruit.x and y == self.fruit.y:
                    state[j+4][i+4] = 1
        
        state = torch.tensor(state).float()
        state = torch.flatten(state)
        
        head_to_fruit = torch.tensor([self.fruit.x-self.snake.x, 
                                self.fruit.y-self.snake.y]).float()

        return torch.cat((state, head_to_fruit))

    def env_step(self, action):
        #takes in an action and puts it through the environment
        self.snake.act(action)
        self.body.update(self.snake.changelist, self.snake)
        if self.collisionfruit():
            self.body.grow(self.snake)
            self.body.update(self.snake.changelist, self.snake)
            self.score += 1
            if self.score == (self.gridlength**2 - 1):
                #incase the agent ever manages to fill the whole grid
                return 100
            self.check_fruit_spawn()
            return self.r_fruit
        elif self.collisionborder():
            return self.r_collision
        elif self.collisionself():
            return self.r_collision
        else:
            return self.r_step

    def env_reset(self):
        #resets the environment and return fresh state
        self.score = 0
        self.snake = Snake(self.snakex, self.snakey)
        self.fruit = Fruit(self.gridlength, 
                            spawnset=self.fruitspawnset, 
                            spawnx=self.fruitx, spawny=self.fruity)
        self.body = Body(0)
        return self.get_state()
    
    def env_store_score(self):
        #used to save a list of scores when the agent is training
        self.scorelist.append(self.score)
        self.score = 0

    
    def render(self, size, textcolor, record=False, path=None):

        #can be used to render the environment
        import pygame
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(((self.gridlength*size),
                                            (self.gridlength*size)))
        pygame.display.init()
        self.screen.fill(self.background)
        for b in self.body.list:
            x = b[0]*size
            y = b[1]*size
            pygame.draw.rect(self.screen, self.bodycolor, 
                                        ((x,y),(size,size)))
        pygame.draw.rect(self.screen, self.snakecolor, 
                    (self.snake.x*size, self.snake.y*size, size, size))
        pygame.draw.rect(self.screen, self.fruitcolor, 
                    (self.fruit.x*size, self.fruit.y*size, size, size))
        pygame.display.update()
        if record:
            pygame.image.save(self.screen,path)

    def save_svg(self,path):
        #can be used to save an svg of the current game state
        d = draw.Drawing(self.gridlength*10, 
                        self.gridlength*10, origin=(0,0))
        b = draw.Rectangle(0,0,self.gridlength*10, 
                            self.gridlength*10, fill='#485B57')
        s = draw.Rectangle(self.snake.x*10,self.snake.y*10,10,10, 
                        fill='#197416',stroke='black', stroke_width=0.02)
        f = draw.Rectangle(self.fruit.x*10,self.fruit.y*10,10,10, 
                        fill='#E92323',stroke='black', stroke_width=0.02)
        d.append(b)
        d.append(s)
        d.append(f)
        for b in self.body.list:
            r = draw.Rectangle(b[0]*10,b[1]*10,10,10, 
                        fill='#32B145',stroke='black', stroke_width=0.02)
            d.append(r)
        d.saveSvg(path)

    def renderreplay(self,gridlength, size, 
                    snakex, snakey, fruitx, fruity, bodylist):
        #can be used to watch an episode of the agent's training history
        import pygame
        screen = pygame.display.set_mode(((gridlength*size),
                                            (gridlength*size)))
        pygame.display.init()
        screen.fill((96,96,96))
        for b in bodylist:
            x = b[0]*size
            y = b[1]*size
            pygame.draw.rect(screen, self.bodycolor, ((x,y),(size,size)))
        pygame.draw.rect(screen, self.snakecolor, 
                                (snakex*size, snakey*size, size, size))
        pygame.draw.rect(screen, self.fruitcolor, 
                                (fruitx*size, fruity*size, size, size))    
        pygame.display.update()

    