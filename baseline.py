###############################################################################
# Course: Evolutionary Computing                                              #  
# Summary: Perform a baseline experiment. Enemy number, number of populations #
# and number of simulations can be set. Results are written to a file named   #
# baseline_enemy_[enemy number].txt                                           #
###############################################################################

import sys, os
import numpy as np

sys.path.insert(0, 'evoman')

from environment import Environment

###############################################################################
################################# Setup #######################################
###############################################################################
enemy_number = 1
n_pop = 50
n_sim = 10
experiment_name = 'baseline_enemy_' + str(enemy_number)

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[enemy_number])

###############################################################################
########################### Baseline Experiment ###############################
###############################################################################
all_avg = []
all_std = []
for sim in range(n_sim):
    fitness = []
    for individual in range(n_pop):
        f, _, _, _ = env.play()
        fitness.append(f)
    avg_pop = np.array(fitness).mean()
    std_pop = np.array(fitness).std()
    all_avg.append(avg_pop)
    all_std.append(std_pop)

###############################################################################
############################# Results #########################################
###############################################################################

result = 'All means:\n' + str(all_avg) + '\nAll STD: \n' + str(all_std) +'\nMean average:\n' + str(np.mean(all_avg)) + '\nSTD average\n' + str(np.mean(all_std))

with open(experiment_name + '.txt', "w") as file:
    file.write(result)
