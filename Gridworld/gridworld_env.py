##############################################################
# This code contains my self-made gridworld environment
##############################################################


import numpy as np
import drawSvg as draw

#inspired by the Gridworld example found in:
#Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.

class Gridworld:
    def __init__(self, m=7, n=14, goal_cords=[3,13], wall_cords_list=[]):
        #initializing grid, player's position, inner walls, 
        #and the goal's position

        self.m = m
        self.n = n
        self.n_actions = 4
        self.game_matrix = np.full((m,n),fill_value=-1)
        self.player_pos = [m//2, 0]
        self.player_pos_temp = [m//2, 0]
        self.wall_cords_list = wall_cords_list
        self.goal_cords = goal_cords
        if len(wall_cords_list) == 0:
            for i in range(6):
                wall_cords_list.append([i,6])
                wall_cords_list.append([6-i,10])


    def move_player(self, action):
        #taking in an action and returning the player's theoratical position
        self.player_pos_temp = self.player_pos.copy()
        if action == 0:
            self.player_pos_temp[0] -= 1
        elif action == 1:
            self.player_pos_temp[1] += 1
        elif action == 2:
            self.player_pos_temp[0] += 1
        elif action == 3:
            self.player_pos_temp[1] -= 1
        else:
            print('invalid action')

    def step(self, action):
        #this function adjusts the environment based on the agent's action 
        #and returns: his new position, reward and a boolean to indicate 
        #whether the episode is done or not

        self.move_player(action)

        #checking whether the agent stepped into an inner wall
        for cords in self.wall_cords_list:
            if cords == self.player_pos_temp:
                reward = -100
                done = True
                self.player_pos = self.player_pos_temp
                return self.player_pos, reward, done

        #checking whether the agent tried to leave the grid
        if (self.player_pos_temp[0] >= self.m or 
            self.player_pos_temp[0] < 0 or 
            self.player_pos_temp[1] >= self.n or 
            self.player_pos_temp[1] < 0):
            reward = -10
            done = False
            return self.player_pos, reward, done
        
        #checking whether the agent reached the goal
        elif self.player_pos_temp == self.goal_cords:
            reward = -1
            done = True
            self.player_pos = self.player_pos_temp
            return self.player_pos, reward, done
        
        #if none of the above check then it was a "standard" action
        else:
            reward = -1
            done = False
            self.player_pos = self.player_pos_temp
            return self.player_pos, reward, done
    
    def reset(self):
        #reseting the agent's position (the environment)
        self.player_pos = [self.m//2, 0]
        self.player_pos_temp = [self.m//2, 0]
        return self.player_pos
    
    def print_matrix(self):
        #for developing purposes
        temp = np.full((self.m,self.n), fill_value=-1)
        temp[tuple(self.goal_cords)] = 0
        for cords in self.wall_cords_list:
            temp[tuple(cords)] = -100
        temp[tuple(self.player_pos)] = 1
        print(temp)
    
    def print_matrix_symbols(self):
        #to quickly check how it is looking and where the agent is located
        #9 = goal, -9 = inner wall, 1 = agent
        temp = np.zeros((self.m,self.n), dtype=int)
        temp[tuple(self.goal_cords)] =  9
        for cords in self.wall_cords_list:
            temp[tuple(cords)] = -9
        temp[tuple(self.player_pos)] = 1
        print(temp)
    
    def draw(self, path, save=True):
        #saves a .svg file of the current environment state

        d = draw.Drawing(142,72, origin=(-1,-1)) #creating canvas

        #drawing inner grid
        for i in range(1,self.m):
            d.append(draw.Line(0,10*i,140,10*i, 
                            stroke='black', stroke_width=0.5)) 
        for i in range(1,self.n):
            d.append(draw.Line(10*i,0,10*i,70, 
                            stroke='black', stroke_width=0.5))
        
        #drawing outer walls
        d.append(draw.Line(0,0,140,0, stroke='black', stroke_width=1))
        d.append(draw.Line(0,10*self.m,140,10*self.m, 
                            stroke='black', stroke_width=1))

        d.append(draw.Line(0,0,0,70, stroke='black', stroke_width=1))
        d.append(draw.Line(10*self.n,0,10*self.n,70, 
                            stroke='black', stroke_width=1))

        #drawing inner walls
        for cords in self.wall_cords_list:
            d.append(draw.Line(10*(cords[1]),10*(self.m-1-cords[0]),
                                10*(cords[1]+1),10*(self.m-1-cords[0]+1), 
                                                        stroke='black'))
            d.append(draw.Line(10*(cords[1]),10*(self.m-1-cords[0]+1),
                                10*(cords[1]+1),10*(self.m-1-cords[0]), 
                                                        stroke='black'))

        #drawing player and goal
        d.append(draw.Circle(self.player_pos[1]*10+5, 
                            (self.m-1-self.player_pos[0])*10+5,3.5, 
                            stroke='blue', fill='white', stroke_width=1))
        d.append(draw.Rectangle(self.goal_cords[1]*10+.75, 
                            (self.m-1-self.goal_cords[0])*10+.75, 8.25, 8.5, 
                            stroke='green', fill='green'))
        if save:
            d.saveSvg(path)
        else:
            return d

    def draw_greedy_policy(self, q_table, path):
        d = self.draw('', save=False)
        arrow = draw.Marker(-0.1, -0.5, 0.9, 0.5, scale=4, orient='auto')
        arrow.append(draw.Lines(-0.1, -0.5, -0.1, 0.5, 0.9, 0, 
                                        fill='red', close=True))


        states = [(i,j) for i in range(self.m) for j in range(self.n)]

        for s in states:
            m = np.argmax(q_table[s])
            optimal_actions = []

            for i,v in enumerate(q_table[s]):
                if v == q_table[s][m]:
                    optimal_actions.append(i)

            if s == (3,0):
                print(optimal_actions)
            
            for o_a in optimal_actions:
                if o_a == 0:
                    d.append(draw.Line(10*(s[1])+5,10*(self.m-1-s[0])+5,
                                    10*(s[1])+5, 10*(self.m-1-s[0])+7, 
                                    stroke='black', stroke_width=0.5, 
                                                    marker_end=arrow))
                elif o_a == 1:
                    d.append(draw.Line(10*(s[1])+5,10*(self.m-1-s[0])+5,
                                    10*(s[1])+7, 10*(self.m-1-s[0])+5, 
                                    stroke='black', stroke_width=0.5, 
                                                    marker_end=arrow))
                elif o_a == 2:
                    d.append(draw.Line(10*(s[1])+5,10*(self.m-1-s[0])+5,
                                    10*(s[1])+5, 10*(self.m-1-s[0])+3, 
                                    stroke='black', stroke_width=0.5, 
                                                    marker_end=arrow))
                elif o_a == 3:
                    d.append(draw.Line(10*(s[1])+5,10*(self.m-1-s[0])+5,
                                    10*(s[1])+3, 10*(self.m-1-s[0])+5, 
                                    stroke='black', stroke_width=0.5, 
                                                    marker_end=arrow))
        d.saveSvg(path)



        
if __name__ == '__main__':

    ###
    #demonstration of how to use the draw_greedy_policy function
    ###

    #initialize environment
    env = Gridworld()

    #load a q-table
    q_table = np.load('q_table.npy', allow_pickle=True)

    #pass the q-table to the function
    #and name the file
    env.draw_greedy_policy(q_table, 'filename.svg')
