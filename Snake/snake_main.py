##############################################################
# This code is a demonstration how to use the SnakeTrainer
# To execute, please place snake_agent.py, snake_env.py,
# snake_trainer.py and snake_main.py in the same directory
##############################################################


import snake_trainer
import numpy as np

#initializing the trainer
trainer = snake_trainer.SnakeTrainer(agent_id=0, updates_per_step=32, 
                    gamma=0.98, lr=0.00005, epsilon=1, epsilon_decay=0.9998, 
                    epsilon_min=0.01, statetype=8, nr_hiddenlayers=1, 
                    nr_neurons=[128], max_mem_length=10000, gridlength=9, 
                    r_fruit=50, r_collision=-50, r_step=0, lr_decay=False, 
                    lr_milestones=[80000], 
                    lr_gamma=0.5, render_bool=False)


#training the agent
trainer.train_n_episodes(n=200,gridlength=3,epsilon_reset=False)
trainer.train_n_episodes(n=400,gridlength=5,epsilon_reset=False)
trainer.train_n_episodes(n=800,gridlength=7,epsilon_reset=False)
trainer.train_n_episodes(n=1600,gridlength=9,epsilon_reset=False)

#watching a replay of his training games
trainer.watch_replay(episodes=[150,2100],gridlengths=[3,9])

#testing the agent
scores = trainer.test_n_episodes(n=10,gridlength=9,render=False,record=False)
print(f'average: {np.average(scores)}')
print(f'high score: {max(scores)}')
