##############################################################
# This code implements the tabular Q-learning algorithm in the
# Cartpole environment provided by OpenAI-Gym
##############################################################

#partly inspired by:
#https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/


import numpy as np
import matplotlib.pyplot as plt
import gym
import math
import multiprocessing as mp
import sys

class CartpoleSolver:
    def __init__(self, obs_high=[2.4, 3.5, 0.21, 1.2], 
                obs_low=[-2.4, -3.5, -0.21, -1.2], 
                obs_chunks=[10, 10, 10, 10],
                q0=0, gamma=1, epsilon=1, alpha=1, epsilon_min=0.1, 
                alpha_min=0.1, decrease_methode=[0,0], S_T_reward=-1000):

        #initializing variables
        self.n_actions = 2
        self.obs_high = np.array(obs_high)
        self.obs_low = np.array(obs_low)
        self.obs_chunks = np.array(obs_chunks)
        self.obs_chunks_delta = (self.obs_high - self.obs_low) / obs_chunks
        self.q0 = q0
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.state_list = []
        self.decrease_methode = decrease_methode
        self.S_T_reward = S_T_reward

        #initializing environment
        self.env = gym.make('CartPole-v0')

    def discretize_state(self, state):
        #turning a continuous state into a descrete one
        discrete_state = list(((state - self.obs_low) / 
                            self.obs_chunks_delta).astype(np.int))
        discrete_state = [min(self.obs_chunks[i]-1, 
                        discrete_state[i]) for i in range(4)]
        return tuple(discrete_state)
    
    def update_q_value(self, q_table, state, new_state, 
                        action, reward, done, alpha):
        #updating the Q-table
        if done:
            #if it is done then the q_table[next_state] 
            #values should all be zero, hence that part is missing
            q_table[state][action] += (alpha * 
                                        (reward - q_table[state][action]))
        else:
            q_table[state][action] += (alpha * 
                                        (reward + 
                                        self.gamma*np.max(q_table[new_state]) - 
                                        q_table[state][action]))
        return q_table
    
    def decrease(self, v, v_min, episode, dm):
        #function to decrease a parameter over time
        #0-3 are different decreasing methods to choose from
        if dm == 0:
            return np.maximum(v_min, 
                        np.minimum(1., 1.0 - math.log10((episode + 1) / 25)))
        elif dm == 1:
            return np.maximum(v_min, v*(0.99)**episode)
        elif dm == 2:
            return np.maximum(v_min, 1.0 / np.sqrt(episode+1))
        elif dm == 3:
            return 1/(episode+1)

    def one_episode(self, epsilon, alpha, render=False):
        #reseting the environment
        done = False
        state = self.discretize_state(self.env.reset())
        score = 0
        while not done:
            if render:
                self.env.render()
            
            #choosing an action based on e-greedy action selection
            action = np.argmax(self.q_table[state])
            if np.random.uniform() < epsilon:
                action = self.env.action_space.sample()

            #taking the selected action and observing its consequences
            new_state, reward, done, _ = self.env.step(action)
            self.state_list.append(new_state)
            new_state = self.discretize_state(new_state)

            #keeping track of the reward
            score += reward

            #adding extra punishment if the agent failed
            if done and score < 200:
                reward = -1000
            
            #updating the q-table
            self.update_q_value(self.q_table, state, new_state, 
                                action, reward, done, alpha)

            state = new_state
        
        return score
    
    def n_episodes(self, n, render=False):
        #initializing q-table, parameters, and score_history 
        #to keep track of the agent's performance
        self.q_table = np.full((tuple(self.obs_chunks)+(self.n_actions,)), 
                                fill_value=self.q0, dtype=float)
        score_history = []
        epsilon = self.epsilon
        alpha = self.alpha

        #playing n episodes and keeping track of the agent's performance
        for e in range(n):
            
            score = self.one_episode(epsilon, alpha, render=render)
            score_history.append(score)
            
            #this would stop the agent and save his q_table 
            #as soon as he reaches the 195 threshold
            #if e > 100:
            #    if np.average(score_history[-100:]) >= 195:
            #        np.save('q_table.npy', self.q_table)
            #        print(e)
            #        sys.exit()

            
            epsilon = self.decrease(epsilon, self.epsilon_min, 
                                    e, self.decrease_methode[0])
            alpha = self.decrease(alpha, self.alpha_min, 
                                    e, self.decrease_methode[1])


        return score_history
    
    def N_n_episodes(self, N, n, render=False):
        #this is to test different sets of parameters, 
        #it takes the average of multiple runs

        s = []

        for _ in range(N):
            s.append(self.n_episodes(n))

        return list(np.average(s, axis=0))

    def plot_runing_average(self, score_histories, labels, title='plot'):
        #plots the average score from the last 100 episodes at each episodes
        for score_history,label in zip(score_histories,labels[1]):
            y = [np.average(score_history[max(0,i-100):i]) 
                    for i in range(len(score_history))] 
            plt.plot(y, label=f'{labels[0]}: {label}', 
                color=f'C{score_histories.index(score_history)}')
            plt.plot(y)
        
        plt.plot([195 for _ in range(len(score_histories[0]))], color='k', 
                        label='195 threshold', linestyle=':')

        plt.legend(loc='lower right', fontsize='x-large')
        plt.xlabel('episode')
        plt.ylabel('average return over last 100 episodes')
        plt.title(title)
        plt.show()
    
    def plot_scores(self, score_histories, labels, title='plot'):
        #plot the score of each episode
        for score_history, label in zip(score_histories, labels[1]):
            plt.plot(score_history, label=f'{labels[0]}: {label}', 
                    color=f'C{score_histories.index(score_history)}')
        plt.legend()
        plt.title(title)
        plt.show()

    def __str__(self):
        #to check the currently used parameters
        return (f'obs_high={self.obs_high}, obs_low={self.obs_low}, ' +
                f'obs_chunks={self.obs_chunks}, q0={self.q0}, ' + 
                f'gamma={self.gamma}, epsilon={self.epsilon}, ' +
                f'alpha={self.alpha}, epsilon_min={self.epsilon_min}, ' +
                f'alpha_min={self.alpha_min}, S_T_reward={self.S_T_reward}, '+
                f'decrease_methode={self.decrease_methode}')



    
    