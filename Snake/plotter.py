##############################################################
# This code can be used to plot different agents' performances
# against each other.
##############################################################

import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

#simply change the folder in which your agents are located
path = os.path.dirname(__file__) + "/final_results"

score_histories = []
labels = []
episodes = []

i = 0
for root, dirs, files in os.walk(path):
    if i == 0:
        i += 1
        pass
    else:
        score = torch.load(root + "/data.tar")['scorelist']
        training_time = torch.load(root + "/data.tar")['main_time']
        episode = torch.load(root + "/data.tar")['round']
        labels.append(root[112:] + f' time: {round(training_time/60)}\'')
        score_histories.append(score)
        episodes.append(episode)

print(labels)
for i, scores in enumerate(score_histories):
    print(f'id: {labels[i]}, max_score: {max(scores)} at e {np.argmax(scores)}, episodes: {episodes[i]}')
    y = [np.average(scores[max(0,i-200):i])
                    for i in range(len(scores))]
    plt.plot(y, label=labels[i], color=f'C{1+i}', linestyle=next(linecycler))

plt.legend()
plt.xlabel("episode")
plt.ylabel("score")
plt.show()

