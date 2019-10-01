import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import glob, os
from deap import base, creator, tools
import random
import numpy as np
from collections import Counter

experiment_name = 'dummy_Kas'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing 
# against static enemy
env = Environment(experiment_name=experiment_name)

n_hidden = 10
n_pop = 60
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 
max_gens = 10

sigma = 1
tau = 1/np.sqrt(n_weights)

offspring_probability = 0.7
mutation_probability = 0.1

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

tlbx = base.Toolbox()
tlbx.register("atrr_float", random.random)
tlbx.register("individual", tools.initRepeat, creator.Individual, 
              tlbx.atrr_float, n = n_weights)
tlbx.register("population", tools.initRepeat, list, tlbx.individual, n = n_pop)

log = tools.Logbook()
pop = tlbx.population()
best = tlbx.individual()

# changes sigma over time
def modify_sigma(tau, sigma=sigma):
    return sigma * np.exp(tau*np.random.normal(0,1))

# mutates alles of gen with p indpb
def self_adaptive_mutate(individual, sigma, indpb):
    mu = 0
    normal_dist = np.random.normal(mu, sigma, len(individual))
    xadd = np.where(np.random.random(normal_dist.shape) < 1-indpb, 0, normal_dist)
    return individual + xadd

# Evaluate 1 individual (i.e. play the game one time)
def evaluate_individual(individual):
    f, _, _, _ = env.play(pcont=individual)
    return f,

def crossover(pop):
    offspring = tlbx.select(pop, int(len(pop)*1.5))
    print('\n\n\n\n\n\n\n\n Attentione!!!!!!')
    print(Counter([child.fitness.values for child in offspring]))
    for parent_1, parent_2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < offspring_probability:   
            tlbx.mate(parent_1, parent_2)
    
    for child in offspring:
        if random.random() < mutation_probability:      
            tlbx.mutate(child, sigma)

    fitness_population = [tlbx.evaluate(child) for child in offspring]
    for individual, fitness in zip(offspring, fitness_population):
        individual.fitness.values = fitness


    return offspring

tlbx.register("evaluate", evaluate_individual)
tlbx.register("mate", tools.cxUniform, indpb=0.1)
tlbx.register('mutate', self_adaptive_mutate, indpb = 0.05)
tlbx.register("select",tools.selTournament, tournsize = 2)
tlbx.register('survival',tools.selTournament, tournsize = 2)

fitness_population = list(map(tlbx.evaluate, pop))

for ind, fit in zip(pop, fitness_population):
    ind.fitness.values = fit

# Evolution
for n_gen in range(max_gens):
    print("Generation {}".format(n_gen+1))

    sigma = modify_sigma(tau, sigma)
    selectionpop = crossover(pop)
    
    fitselect = [ind.fitness.values[0] for ind in selectionpop]

    pop[:] = tlbx.survival(selectionpop, len(pop))
    
    fits = [ind.fitness.values[0] for ind in pop]

    maxval = np.max(fits)
    index = fits.index(maxval)
    log.record(gen = n_gen, meanfit = np.mean(fits), varfit = np.var(fits), 
               stdfit = np.std(fits), maxfit = maxval, 
               optweightcombination = pop[index])

    if best.fitness.valid != True or best.fitness.values <= pop[index].fitness.values:
        best = tlbx.clone(pop[index])

print(best.fitness.values)