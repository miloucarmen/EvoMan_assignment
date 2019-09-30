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

experiment_name = 'dummy_bart'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
				  enemies=[2])

env.state_to_log() # checks environment state


random.seed(1)

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f, p, e, t

# evaluation
def evaluate(pop):
    return np.array(list(map(lambda y: simulation(env,y), pop)))

n_hidden = 10
pop_size = 5
ngens = 10
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
min_allele = -1
max_allele = 1


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

tlbx = base.Toolbox()

tlbx.register("atrr_float", np.random.random)
tlbx.register("individual", tools.initRepeat, creator.Individual, tlbx.atrr_float, n=n_weights)
tlbx.register("Population", tools.initRepeat, list, tlbx.individual, n=pop_size)
tlbx.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=(1/n_weights))
tlbx.register("evaluate", evaluate)


pop = tlbx.Population()

for g in range(ngens):
    print("Generation: {}".format(g+1))

    # Apply mutation on the offspring
    pm = 1/pop_size
    for mutant in pop:
        if random.random() < pm:
            tlbx.mutate(mutant)
            del mutant.fitness.values

    new_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitns = list(map(tlbx.evaluate, new_ind))
    newpopfit = tlbx.evaluate(pop)
    print(sum(newpopfit[:,0])/pop_size)
