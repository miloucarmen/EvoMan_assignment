##########################
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

random.seed(1)

experiment_name = 'dummy_Alex'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name, enemies = [2],  speed="fastest")

n_hidden = 10
n_pop = 10
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
max_gens = 1
n_gen = 0

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

tlbx = base.Toolbox()
tlbx.register("atrr_float", random.random)
tlbx.register("individual", tools.initRepeat, creator.Individual, tlbx.atrr_float, n = n_weights)
tlbx.register("Population", tools.initRepeat, list, tlbx.individual, n = n_pop)

log = tools.Logbook()
Pop = tlbx.Population()


# evaluation
def EvaluateFit(individual):
    f,p,e,t = env.play(pcont=individual )
    return f,

def Normalise(fit, fitnesses):

    if ( max(fitnesses) - min(fitnesses) ) > 0 :
        fitnorm = ( fit - min(fitnesses) )/( max(fitnesses) - min(fitnesses) )

    else :
        fitnorm = 0

    if fitnorm < 0:
            fitnorm = 0.0000000001
    return fitnorm

# Can be used for uniform parent selection

def uniform_parent(pop): # the pop the portion of total pop you want as chosen individuals

    """the selection for the 'mating population' is created by uniform distribution
    and is 3 times the size of the orginial population"""
    chosen_ind = []
    len_matingpop = 3 * len(pop)

    for ind in range(0, len_matingpop):
        num = random.randint(0, (len(pop)-1))
        chosen_ind.append(pop[num])

    return chosen_ind



tlbx.register("evaluate", EvaluateFit)
tlbx.register("mate", tools.cxTwoPoint)
tlbx.register('mutate', tools.mutFlipBit, indpb = 0.05)
tlbx.register("select",tools.selTournament, tournsize = 3)
tlbx.register('survival',tools.selTournament, tournsize = 3 )
tlbx.register('uniform_parents', uniform_parent)

OffProb, MuProb = 0.5, 0.1
# tlbx.register("normialise", Normalise)

# evaluate initial pop

fitns = list(map(tlbx.evaluate, Pop))
# fitnsnorm = list(map(lambda x: tlbx.normialise(x, fitns), fitns))
# print(fitnsnorm)

for ind, fit in zip(Pop, fitns):
    # print(fit)
    ind.fitness.values = fit

fit = [ind.fitness.values[0] for ind in Pop]

while max(fit) < 100 and n_gen < max_gens:

    print("---------------------Generation %i-------------------------", n_gen)

    # parent selection

    # This is tournament as parent selection
    # offspring = tlbx.select(Pop, len(Pop)//2) # half the population size
    # offspring = list(map(tlbx.clone, offspring))

    offspring = tlbx.uniform_parents(Pop)
    offspring = list(map(tlbx.clone, offspring))
    originalpop = list(map(tlbx.clone, Pop))


    # for selected parents, decide who is going to mate
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < OffProb:

            tlbx.mate(child1,child2) #we expect 1/4 of the population
            # returns a tuple of two individuals

            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MuProb:

            tlbx.mutate(mutant)
            del mutant.fitness.values

    new_ind = []
    for ind in offspring:
        if ind.fitness.valid != True:
            new_ind.append(ind)

    # new_ind = [ind for ind in offspring if not ind.fitness.valid]
    # the fitness of the new individuals

    fitns = list(map(tlbx.evaluate, new_ind))


    for ind, fit in zip(new_ind, fitns):
        ind.fitness.values = fit

    selectionpop = originalpop + new_ind
    fitselect = [ind.fitness.values[0] for ind in selectionpop]

    Pop[:] = tlbx.survival(selectionpop, len(Pop))

    fits = [ind.fitness.values[0] for ind in Pop]

    maxval = np.max(fits)
    index = fits.index(maxval)
    log.record(gen = n_gen, meanfit = np.mean(fits), varfit = np.var(fits), stdfit = np.std(fits), maxfit = maxval, optweightcombination = Pop[index])
    n_gen += 1
