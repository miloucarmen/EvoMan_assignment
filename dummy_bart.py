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
				  enemies=[1])

env.state_to_log() # checks environment state


random.seed(1)

# standard variables
n_hidden = 10
pop_size = 5
ngens = 10
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5

# sigma for normal dist, tao constant,  pm prop mutation for individu
sigma = 1
tao = 1/np.sqrt(n_weights)
pm = 1/pop_size


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f, p, e, t

# evaluation
def evaluate(pop):
    return np.array(list(map(lambda y: simulation(env,y), pop)))

# changes sigma over time
def modify_sigma(tao, sigma=sigma):
    return sigma * np.exp(tao*np.random.normal(0,1))

# mutates alles of gen with p indpb
def self_adaptive_mutate(individual, sigma, indpb):
    mu = 0
    print('prob allel:', indpb)
    print('newsig:',sigma)
    normal_dist = np.random.normal(mu, new_sigma, len(individual))
    xadd = np.where(np.random.random(normal_dist.shape) < 1-indpb, 0, normal_dist)
    return individual + xadd

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

tlbx = base.Toolbox()

tlbx.register("atrr_float", np.random.random)
tlbx.register("individual", tools.initRepeat, creator.Individual, tlbx.atrr_float, n=n_weights)
tlbx.register("Population", tools.initRepeat, list, tlbx.individual, n=pop_size)

tlbx.register("mutate", self_adaptive_mutate, indpb=(1/n_weights))
tlbx.register("evaluate", evaluate)

pop = tlbx.Population()

for g in range(ngens):
    print("Generation: {}".format(g + 1))

    # Apply mutation on the offspring
    new_sigma = modify_sigma(tao, sigma=sigma)
    for mutant in pop:
        if random.random() < pm:
            tlbx.mutate(mutant, new_sigma)
            del mutant.fitness.values
            sigma = new_sigma

    newpopfit = tlbx.evaluate(pop)
    print(sum(newpopfit[:,0])/pop_size)
