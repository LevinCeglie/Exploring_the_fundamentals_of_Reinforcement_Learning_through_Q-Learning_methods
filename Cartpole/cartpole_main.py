##############################################################
# This code is a demonstration how to use the CartpoleSolver
# To execute, please place cartpole_solver.py and
# cartpole_main.py in the same directory
##############################################################


import cartpole_solver
import numpy as np

#to keep track of the scores
score_histories = []

#at index 0 is the parameter that is being adjusted as str
#at index 1 is a list of different values to be tested for said parameter
parameters_to_be_tried = ['alpha',[0.1,0.1,0.1,0.1,0.1]]

#iterates through the values to be tried
for p in parameters_to_be_tried[1]:
    solver = cartpole_solver.CartpoleSolver(
                        obs_high=[2.4, 2, 0.21, 1.8], 
                        obs_low=[-2.4, -2, -0.21, -1.8], 
                        obs_chunks=[1, 1, 12, 12], q0=0, 
                        gamma=1, epsilon=1, alpha=p, 
                        epsilon_min=0.01, alpha_min=p, 
                        decrease_methode=[3,0], S_T_reward=-2000)
    #(amount of agents, amount of episodes) to be tested on a value
    l = solver.N_n_episodes(1,500)
    score_histories.append(l)
    print(solver.__str__())

#setting the title for the plot
title = f'adjusting {parameters_to_be_tried[0]}'

#saving the data (optional)
#so it can be laoded later to plot it again
np.save(f'data_{parameters_to_be_tried[0]}.npy', 
        np.array([score_histories, parameters_to_be_tried, title], 
                                                dtype=object))

#plotting the results
solver.plot_runing_average(score_histories, 
                            parameters_to_be_tried, title=title)

###
#example of loading data and plotting
###

#initializing the CartpoleSolver object containing plotter function
plotter = cartpole_solver.CartpoleSolver()

#laoding the data
data = np.load('data_[name].npy', allow_pickle=True)

#setting the title and plotting the data
title = f'adjusting {data[1][0]}'
plotter.plot_runing_average(data[0],data[1],title)


###
#alternatively one can test a single specific set of parameters
###

#initializing solver with specific parameters
solver = cartpole_solver.CartpoleSolver(
                obs_high=[2.4, 2, 0.21, 1.8], 
                obs_low=[-2.4, -2, -0.21, -1.8], 
                obs_chunks=[1, 1, 12, 12], q0=0, 
                gamma=1, epsilon=1, alpha=0.9, 
                epsilon_min=0.01, alpha_min=0.9, 
                decrease_methode=[3,0], S_T_reward=-2000)

#either by a single agent (n = amounts of episodes)
score_history = solver.n_episodes(n=1000,render=False)

#or by many agents (N = amount of agents)
score_histories = solver.N_n_episodes(N=10,n=1000,render=False)