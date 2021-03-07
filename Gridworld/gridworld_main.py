##############################################################
# This code is a demonstration how to use the GridworldSolver
# (utilizing multiprocessing to speed it up)
# To execute, please place gridworld_env.py, gridworld_solver.py
# and gridworld_solver.py in the same directory
##############################################################


import gridworld_solver
import multiprocessing as mp
import time
import numpy as np

#here one can change the parameters other than the one being adjusted
#every parameter other than the one being adjusted must have a default value
def one_process(epsilon, q0=0, gamma=1, alpha=.5, N=100, n=20000):
    solver = gridworld_solver.GridworldSolver(q0, gamma, epsilon, alpha)
    return solver.N_n_episodes(N,n)

#this is needed in order for the multiprocessing to work
if __name__ == '__main__':

    #keeping track of important metrics
    s0 = time.time()
    score_histories = []
    solved_after = []

    #at index 0 is the parameter that is being adjusted as str
    #at index 1 is a list of different values to be tested for said parameter
    parameters = ['epsilon', [0.1,0.3,0.5,0.7]]

    #using multiprocessing to 
    p = mp.Pool(mp.cpu_count())
    l = p.map(one_process,parameters[1])
    for game in l:
        score_histories.append(game[0])
        solved_after.append(game[1])

    #saving the data
    np.save(f'data_{parameters[0]}.npy', np.array([score_histories, 
                                solved_after, parameters], dtype=object))

    plotter = gridworld_solver.GridworldSolver()
    #the tile has to be adjusted manually
    plotter.plot_scores(score_histories, parameters, solved_after, 
                        title='epsilon=_, alpha=0.5, q0=0, gamma=1')

    #to see how long it took                    
    print(f'calculations took {round((time.time()-s0)/60)} minutes')

###
#if one does not want to compare different values for a parameter
#and would simply like to try one specific set of parameters
#here is how to do it
###

#set parameters and create solver object
solver = gridworld_solver.GridworldSolver(epsilon=0.3,alpha=0.9, q0=-90)
#play a specific amount of episodes
s, solved = solver.n_episodes(20000)
#save the q-table (optional)
np.save('q_table.npy', solver.q_table)
