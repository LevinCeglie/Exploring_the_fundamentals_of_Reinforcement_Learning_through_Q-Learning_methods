##############################################################
# This code comibines the Snake environment and the DQL agent
# It combines them in a python object, which can be used to
# train an agent with freely adjustable parameters
##############################################################


import snake_agent
import snake_env
import os
import torch
import time
import sys
import numpy as np

class SnakeTrainer:
    def __init__(self, agent_id, updates_per_step, gamma, lr, 
                epsilon, epsilon_decay, epsilon_min, statetype, 
                nr_hiddenlayers, nr_neurons, max_mem_length, 
                gridlength, r_fruit, r_collision, r_step, 
                lr_decay=False, lr_milestones=None, lr_gamma=None, 
                render_bool=False):

        #selecting device, preferably the gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                            else 'cpu')

        #paths for data collection
        self.agent_id = agent_id
        self.path = (os.path.dirname(os.path.abspath(__file__)) 
                    + f'/results/{agent_id}/')

        self.modelpath = self.path + 'checkpoint.tar'
        self.settingspath = self.path + 'settings.txt'
        self.datapath = self.path + 'data.tar'
        self.memorypath = self.path + 'memory.tar'
        self.replaypath = self.path + 'replay.npz'

        #initializing important variables
        self.updates_per_step = updates_per_step
        self.render_bool = render_bool

        #initializing the environment
        self.env = snake_env.Environment(gridlength=gridlength, 
            snakex=gridlength//2, snakey=gridlength//2, 
            fruitspawnset=False, r_fruit = r_fruit, 
            r_collision = r_collision, r_step = r_step, 
            statetype=statetype)

        self.statetype = statetype
        self.rewards_fcs = [r_fruit,r_collision,r_step]

        obs_space = len(self.env.env_reset())

        #initializing the agent
        self.agent = snake_agent.Agent(gamma=gamma, lr=lr, epsilon=epsilon, 
            epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, 
            obs_space=obs_space, act_space=4, device=self.device, 
            updates_per_step=self.updates_per_step, lr_decay=lr_decay, 
            lr_milestones=lr_milestones, lr_gamma=lr_gamma,
            max_mem_length=max_mem_length, 
            nr_hiddenlayers=nr_hiddenlayers, nr_neurons=nr_neurons)

        #initializing data collection
        if not os.path.isdir(self.path):
            self.init_data()
        else:
            self.load_agent()


    def init_data(self):
        #creates a directory and a first textfile with agent's specs
        os.makedirs(self.path)
        textfile = open(self.settingspath, 'a')
        textfile.write(f"agent_id: {self.agent_id}"
                    +f"device: {self.device} \nDQN: \n{self.agent.dqn}\n"
                    +f"\noptimizer: \n{self.agent.optimizer}\n"
                    +f"\ngamma: {self.agent.gamma} \nepsilon_decay & _min: "
                    +f"{self.agent.epsilon_decay}, {self.agent.epsilon_min}"
                    +f"\nstatetype: {self.env.statetype} \nupdates_per_step: "
                    +f"{self.agent.updates_per_step} \nmax_mem_length: "
                    +f"{self.agent.max_mem_length} \nrewards(f,c,s): "
                    +f"{self.env.r_fruit}, {self.env.r_collision}, "
                    +f"{self.env.r_step} \nlr_decay: {self.agent.lr_decay}")
        if self.agent.lr_decay:
            textfile.write(f"lr_scheduler: \n{self.agent.scheduler}\n"
                    +f"lr_milestones: {self.agent.lr_milestones} \nlr_gamma: "
                    +f"{self.agent.lr_gamma}")
        textfile.close()

        #since its the first time this agent palys
        # new data collection lists are created
        self.episode_nr = 0
        self.epsilon_history = []
        self.stime_history = []
        self.q_history = []
        self.replay_memory = []
        self.replay_memorytemp = []
        self.updates = 0
        self.main_time = 0
 
    def load_agent(self):
        #laods all that is necessary
        
        #loading model
        checkpoint = torch.load(self.modelpath)
        self.agent.dqn.load_state_dict(checkpoint['model_state_dict'])
        self.agent.optimizer.load_state_dict(
                                    checkpoint['optimizer_state_dict'])
        if self.agent.lr_decay:
            self.agent.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("model was loaded")

        #loading data
        checkpoint = torch.load(self.datapath)
        self.epsilon_history = checkpoint['epsilon_history']
        self.stime_history = checkpoint['sruvival_time_history']
        self.q_history = checkpoint['q_history']
        self.env.scorelist = checkpoint['scorelist']
        self.episode_nr = checkpoint['episode_nr']
        self.agent.epsilon = checkpoint['epsilon']
        self.updates = checkpoint['updates']
        self.main_time = checkpoint['main_time']
        print("data was loaded")

        #loading agent's memory
        self.agent.memory = torch.load(self.memorypath)
        print("memory was loaded")

        #loading replay memory
        self.replay_memorytemp = []
        if os.path.exists(self.replaypath):
            self.replay_memory = np.load(self.replaypath, 
                            allow_pickle=True)['replay'].tolist()
        print("replay memory was loaded")

    def save_model(self):
        #saves the model
        if self.agent.lr_decay:
            torch.save({
                'model_state_dict': self.agent.dqn.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'lr_scheduler': self.agent.scheduler.state_dict(),
                }, self.modelpath)
        else:
            torch.save({
                'model_state_dict': self.agent.dqn.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict()
                }, self.modelpath)
    
    def save_data(self):
        #saves the data
        torch.save({
            'episode_nr': self.episode_nr,
            'epsilon': self.agent.epsilon,
            'epsilon_history': self.epsilon_history,
            'sruvival_time_history': self.stime_history,
            'q_history': self.q_history,
            'scorelist': self.env.scorelist,
            'updates': self.updates,
            'main_time': self.main_time
            }, self.datapath)
    
    def save_memory(self):
        #save the memory
        torch.save(self.agent.memory,self.memorypath)
    
    def save_replay(self):
        np.savez_compressed(self.replaypath,
                    replay=np.array(self.replay_memory, dtype=object))


    def train_n_episodes(self, n, gridlength, epsilon_reset=False):
        #loading new environment incase the gridlength changed
        self.env = snake_env.Environment(gridlength=gridlength, 
            snakex=gridlength//2, snakey=gridlength//2, 
            fruitspawnset=False, r_fruit = self.rewards_fcs[0], 
            r_collision = self.rewards_fcs[1], r_step = self.rewards_fcs[2], 
            statetype=self.statetype)
    
        if os.path.exists(self.datapath):
        	self.load_agent()

        if epsilon_reset:
            self.agent.epsilon = 1
        
        #starting the main for loop to play n episodes
        for e in range(n):
            self.episode_nr += 1
            self.env.env_reset()
            current_body = []
            for b in self.env.body.list:
                current_body.append(b)
            self.replay_memorytemp.append((self.env.snake.x, 
                    self.env.snake.y, self.env.fruit.x, 
                    self.env.fruit.y, current_body))
            
            main_time_temp = time.time()

            for i in range(1500):
                #receiving the observation from the environment
                #choosing an action and take it
                #receiving reward
                state = self.env.get_state()
                state = state.to(self.device)
                action = self.agent.act(state)
                reward = self.env.env_step(action)
                
                #keeping track of the estimated total reward at s_0
                if i == 0:
                    self.q_history.append(torch.max(
                                    self.agent.dqn.forward(state)).item())

                if self.render_bool:
                    self.env.render(60, (0,0,0))

                #saving the frame for later replay
                current_body = []
                for b in self.env.body.list:
                    current_body.append(b)
                self.replay_memorytemp.append((self.env.snake.x, 
                            self.env.snake.y, self.env.fruit.x, 
                                self.env.fruit.y, current_body))
                

                #checking if he collided or fille the whole grid
                if reward == self.rewards_fcs[1]:
                    done = True
                elif reward == 100:
                    #incase the agent ever manages to fill the whole grid
                    done = True
                else:
                    done = False

                #receiving new state
                new_state = self.env.get_state()
                new_state = new_state.to(self.device)

                #saving the agent's experience
                self.agent.savedata(state, action, 
                                    reward, new_state, done)

                #as soon as there is enough data the training will start
                if len(self.agent.memory) > self.updates_per_step:
                    self.agent.train()
                    self.updates += self.updates_per_step

                #if he collided with either the wall or himself, 
                # the episode is over
                if done:            
                    break
            
            #keeping track of important metrics
            self.epsilon_history.append(self.agent.epsilon)
            self.env.env_store_score()
            self.stime_history.append(i+1)
            print(f"episode: {self.episode_nr}"
                +f" score: {self.env.scorelist[-1]}"
                +f" surv_time: {i+1}"
                +f" epsilon: {round(self.agent.epsilon,3)}"
                +f" q: {round(self.q_history[-1])}")


            #keeping track of time taken
            self.main_time += time.time()-main_time_temp

            self.replay_memory.append(self.replay_memorytemp)
            self.replay_memorytemp = []

            #every 50th episode everything is saved
            if self.episode_nr%50 == 0:
                print("saving...")
                self.save_model()
                self.save_data()
                self.save_memory()
                self.save_replay()
                print("done saving!")

    def test_n_episodes(self, n, gridlength, record=False, render=True):
        #loading new environment incase the gridlength changed
        #recort = True will save a .png of every frame
        self.env = snake_env.Environment(gridlength=gridlength, 
            snakex=gridlength//2, snakey=gridlength//2, 
            fruitspawnset=False, r_fruit = self.rewards_fcs[0], 
            r_collision = self.rewards_fcs[1], r_step = self.rewards_fcs[2], 
            statetype=self.statetype)
    
        if os.path.exists(self.datapath):
        	self.load_agent()

        score_list = []

        import pygame
        clock = pygame.time.Clock()
        for e in range(n):
            self.env.env_reset()
            rewards = []
            for i in range(1500):
                if render:
                    clock.tick(10)
                #receiving the observation from the environment
                #choosing an action and take it
                #receiving reward
                state = self.env.get_state()
                state = state.to(self.device)
                action = self.agent.act_greedy(state)
                reward = self.env.env_step(action)
                rewards.append(reward)
                
                #keeping track of the estimated total reward at s_0
                if i == 0:
                    q0 = torch.max(self.agent.dqn.forward(state)).item()

                if render:
                    self.env.render(60, (0,0,0), record=record, 
                    path=self.path + f"frame{i}.png")


                #checking if he collided or fille the whole grid
                if reward == self.rewards_fcs[1]:
                    done = True
                elif reward == 100:
                    done = True
                else:
                    done = False

                #receiving new state
                new_state = self.env.get_state()
                new_state = new_state.to(self.device)

                #if he collided with either the wall or himself, 
                # the episode is over
                if done:            
                    break
            
            real_q0 = 0
            for j in range(len(rewards)):
                real_q0 += rewards[j] * (self.agent.gamma**(j))

            print(f'score: {self.env.score}, surv_time: {i},'
                + f' est q0: {round(q0)}, real q0: {round(real_q0)}')
            score_list.append(self.env.score)
        return score_list
    
    def watch_replay(self, episodes, gridlengths):
        #used to watch a replay of a specific training episode
        import pygame
        pygame.init()
        clock = pygame.time.Clock()
        for e,g in zip(episodes,gridlengths):
            print(self.env.scorelist[e])
            for snakex,snakey,fruitx,fruity,bodylist in self.replay_memory[e]:
                clock.tick(10)
                self.env.renderreplay(gridlength=g, size=45, 
                                snakex=snakex, snakey=snakey, 
                                fruitx=fruitx, fruity=fruity, 
                                            bodylist=bodylist)
