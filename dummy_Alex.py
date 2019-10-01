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

# initializes environment with ai player using random controller, playing against static enemy nr 2
env = Environment(experiment_name=experiment_name, enemies = [2],  speed="fastest")

# define population parameters
n_hidden = 10
n_pop = 10
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
max_gens = 1
n_gen = 0
noimprovement = 0

# sigma for normal dist, tao constant,  pm prop mutation for individu
sigma = 1
tao = 1./np.sqrt(n_weights)
pm = 1./n_pop
OffProb, MuProb = 0.5, 0.1

# initialse individual and population creator
creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

# register the function
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

# changes sigma over time
def modify_sigma(tau, sigma=sigma):
    return sigma * np.exp(tau*np.random.normal(0,1))


def self_adaptive_mutate(individual, sigma, indpb):
    mu = 0
    print('prob allel:', indpb)
    print('newsig:',sigma)
    normal_dist = np.random.normal(mu, sigma, len(individual))
    xadd = np.where(np.random.random(normal_dist.shape) < 1-indpb, 0, normal_dist)
    return individual + xadd

def natural_selection(selectionpop, pop_size):
    fitselect = [ind.fitness.values[0] for ind in selectionpop]
    pop = []
    for _ in range(pop_size):
        idx_inds = random.sample(range(len(fitselect)), 3)
        fitness_inds = np.array(fitselect)[idx_inds]
        best_idx = idx_inds[np.argmax(fitness_inds)]
        pop.append(selectionpop.pop(best_idx))
        fitselect.pop(best_idx)
    return pop

def uniform_parent(pop):

    """the selection for the 'mating population' is created by uniform distribution
    and is 3 times the size of the orginial population"""

    chosen_ind = []
    len_matingpop = 3 * len(pop)

    for ind in range(0, len_matingpop):
        num = random.randint(0, (len(pop)-1))
        chosen_ind.append(pop[num])

    return chosen_ind


# define the toolbox functions

tlbx.register("evaluate", EvaluateFit)
tlbx.register("mate", tools.cxTwoPoint)
tlbx.register('mutate', tools.mutFlipBit, indpb = sigma)
tlbx.register("select",tools.selTournament, tournsize = 3)
tlbx.register('survival',tools.selTournament, tournsize = 3 )
tlbx.register('uniform_parents', uniform_parent)
tlbx.register("normalise", Normalise)
# last three functions need to be provided

# probability on offspring and mubrob

# evaluate initial pop
fitns = list(map(tlbx.evaluate, Pop))

for ind, fit in zip(Pop, fitns):
    # print(fit)
    ind.fitness.values = fit

fit = [ind.fitness.values[0] for ind in Pop]

# maxval = np.max(fit)
# index = fit.index(maxval)

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
            tlbx.mate(child1,child2)
            del child1.fitness.values
            del child2.fitness.values

    sigma = modify_sigma(tao, sigma=sigma)

    for mutant in offspring:
        if random.random() < pm:
            tlbx.mutate(mutant)
            del mutant.fitness.values

    new_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitns = list(map(tlbx.evaluate, new_ind))

    for ind, fit in zip(new_ind, fitns):
        ind.fitness.values = fit

    Pop[:] = tlbx.survival(offspring, len(Pop))

    fits = [ind.fitness.values[0] for ind in Pop]
    log.record(gen = n_gen, meanfit = np.mean(fits), varfit = np.var(fits), stdfit = np.std(fits), maxfit =  np.max(fits), optweightcombination = Pop[fits.index(np.max(fits))])

    n_gen += 1
    # if best.fitness.valid != True or best.fitness.values <= Pop[index].fitness.values:
    #     best = tlbx.clone(Pop[index])
    #
    # if log[n_gen].get("meanfit") <= log[n_gen - 1].get("meanfit") :
    #     noimprovement += 1
    # else :
    #     noimprovement = 0
    #
    # if noimprovement > 1:
    #     print('~~~~~~~~~~DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOM~~~~~~~~~~')
    #     del log[n_gen]
    #     Doomsday(Pop)
    #     fits = [ind.fitness.values[0] for ind in Pop]
    #     log.record(gen = n_gen, meanfit = np.mean(fits), varfit = np.var(fits), stdfit = np.std(fits), maxfit =  np.max(fits), optweightcombination = Pop[fits.index(np.max(fits))])

print(log.select("meanfit"))
print(best.fitness.values)
