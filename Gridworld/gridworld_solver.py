##############################################################
# This code implements the tabular Q-learning algorithm in a
# self-made gridworl environment (described in Section 3.2)
##############################################################


import numpy as np
import matplotlib.pyplot as plt
from gridworld_env import Gridworld
import math

#partly inspired by: 
#https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/

class GridworldSolver:
    def __init__(self, q0=0, gamma=1, epsilon=0.1, alpha=0.1):
        #initializing the gridworld, the parameters and other 
        #later useful variables
        self.env = Gridworld()
        
        self.n_actions = self.env.n_actions
        self.q0 = q0
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        self.optimal_s0 = [-27, -25, -25, -35]
        
    def init_q_table(self):
        #initializing q-table with user-inputed value
        self.q_table = np.full(((self.env.m, self.env.n, self.n_actions,)), 
                                            fill_value=self.q0, dtype=float)
        
        #setting q-value of terminal states to 0
        self.q_table[tuple(self.env.goal_cords)] = [0 for _ in range(self.n_actions)]
        for cords in self.env.wall_cords_list:
            self.q_table[tuple(cords)] = [0 for _ in range(self.n_actions)]


    def update_q_value(self, state, new_state, action, reward, done, alpha):
        #function to update the q-table based on observations
        self.q_table[state][action] += (alpha * (reward + self.gamma * 
                                        np.max(self.q_table[new_state]) - 
                                        self.q_table[state][action]))

    def one_episode(self, epsilon, alpha):
        #reseting the environment
        done = False
        state = tuple(self.env.reset())
        score = 0
        while not done:
            #choosing an action based on e-greedy action selection
            action = np.argmax(self.q_table[state])
            if np.random.uniform() < epsilon:
                action = np.random.randint(0,self.n_actions)


            
            #taking the selected action and observing its consequences
            new_state, reward, done = self.env.step(action)
            new_state = tuple(new_state)

            #keeping track of the reward
            score += reward
            
            #updating the q-table
            self.update_q_value(state, new_state, action, reward, done, alpha)

            state = new_state
        
        return score
    
    def n_episodes(self, n):
        #initializing q-table, parameters, and score_history 
        #to keep track of the agent's performance
        self.init_q_table()

        score_history = []
        solved_after = -1
        epsilon = self.epsilon
        alpha = self.alpha

        #playing n episodes and keeping track of the agent's performance
        for e in range(n):
            
            score = self.one_episode(epsilon, alpha)
            score_history.append(score)

            ###
            #uncomment the following, if one would like to save the q-table
            #at the point were it was "optimal" for the first time
            #"optimal" as described in Section 3.2
            ###

            #if (list(self.q_table[(3,0)]) == self.optimal_s0 
            #                        and solved_after == -1):
            #    np.save('q_table.npy', self.q_table)
            #    solved_after = e

        return score_history, solved_after
    
    def N_n_episodes(self, N, n):
        #this is to test different sets of parameters, 
        #it takes the average of multiple runs
        s = []
        solved_average = []

        for _ in range(N):
            l = self.n_episodes(n)
            s.append(l[0])
            solved_average.append(l[1])

        return list(np.average(s, axis=0)), np.average(solved_average)
    
    def plot_scores(self, score_histories, labels, solved_after, title='plot'):
        #plots the agent's score of the individual episodes
        #the episode on which he solved 

        plt.rcParams.update({'font.size': 15})
        for i,score_history in enumerate(score_histories):
            plt.plot(score_history, label=f'{labels[0]}: {labels[1][i]}', 
                    color=f'C{i}')
            if solved_after[i] != -1:
                plt.plot([solved_after[i],solved_after[i]],[-50,0], 
                        color=f'C{i}', linestyle='--', linewidth=3, 
                        label=f'converged after {solved_after[i]} episodes')

        plt.plot([-25 for _ in range(len(score_histories[0]))], color='k', 
                                                label='optimal return, -25')
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.legend(loc='lower right', fontsize='x-large')
        plt.title(title)
        plt.show()


    def __str__(self):
        #to check the currently used parameters
        return (f'epsilon={self.epsilon}, alpha={self.alpha}, '+
                f'q0={self.q0}, gamma={self.gamma}')

