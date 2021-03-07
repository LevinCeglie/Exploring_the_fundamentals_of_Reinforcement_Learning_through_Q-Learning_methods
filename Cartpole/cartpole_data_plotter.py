##############################################################
# This code to load and plot the data save in cartpole_main.py
# to execute please place cartpole_solver.py, 
# cartpole_data_plotter.py and the data.npy in the same directory
##############################################################


import cartpole_solver
import numpy as np

#initializing the CartpoleSolver object containing plotter function
plotter = cartpole_solver.CartpoleSolver()

#laoding the data
data = np.load('data_[name].npy', allow_pickle=True)

#setting the title and plotting the data
title = f'adjusting {data[1][0]}'
plotter.plot_runing_average(data[0],data[1],title)
