################################
#                              #
################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import glob, os
from deap import base, creator, tools
import random
import numpy as np

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)

# env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

# ini = time.time()  # sets time marker


# genetic algorithm params

# run_mode = 'train' # train or test


n_hidden = 10
Npop = 10
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

random.seed(1)

tlbx = base.Toolbox()
tlbx.register("atrr_float", random.random)
tlbx.register("individual", tools.initRepeat, creator.Individual, tlbx.atrr_float, n = n_weights)
tlbx.register("Population", tools.initRepeat, list, tlbx.individual, n = Npop)

Pop = tlbx.Population()
 

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

f = evaluate(Pop)
