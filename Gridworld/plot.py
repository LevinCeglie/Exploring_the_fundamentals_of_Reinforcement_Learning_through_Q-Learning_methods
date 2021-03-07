import numpy as np
from gridworld_solver import GridworldSolver

plotter = GridworldSolver()

a = np.load('data_q02.npy', allow_pickle=True)

plotter.plot_scores(a[0], a[2], a[1], title='epsilon=0.3, alpha=0.9, q0=_, gamma=1')