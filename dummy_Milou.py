################################
#                              #
################################
# groetjes kas
# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import glob, os
from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(1)

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name, enemies = [2])

n_hidden = 10
n_pop = 10
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 
max_gens = 50
n_gen = 0
noimprovement = 0
low_bound = -1
upper_bound = 1

# sigma for normal dist, tao constant,  pm prop mutation for individu
sigma = 1
tau = 1/np.sqrt(n_weights)
mut_prob = 1/n_pop


creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

tlbx = base.Toolbox()
tlbx.register("atrr_float", random.uniform, low_bound, upper_bound)
tlbx.register("individual", tools.initRepeat, creator.Individual, tlbx.atrr_float, n = n_weights)
tlbx.register("Population", tools.initRepeat, list, tlbx.individual, n = n_pop)


log = tools.Logbook()
Pop = tlbx.Population()
best = tlbx.individual()


# evaluation
def EvaluateFit(individual):
    f,p,e,t = env.play(pcont=individual)
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

# mutates alles of gen with p indpb
def self_adaptive_mutate(individual, sigma, indpb):
    
    mu = 0
    normal_dist = np.random.normal(mu, sigma, len(individual))
    xadd = np.where(np.random.random(normal_dist.shape) < 1-indpb, 0, normal_dist)
    individual = individual + xadd
    for i in range(len(individual)):
        if individual[i] > upper_bound:
            individual[i] = upper_bound
        elif individual[i] < low_bound:
            individual[i] = low_bound

    return individual

# natural selection of population, without replacement
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

def Doomsday(pop, fit, sigma):
    # replaces 25% of population
    worst = int(n_pop/4)  # a quarter of the population
    order = np.argsort(fit)
    orderasc = order[0:worst]

    for i in orderasc:
        self_adaptive_mutate(pop[i], sigma, indpb=0.05)
        newfit = tlbx.evaluate(pop[i])
        pop[i].fitness.values = newfit

    return pop

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
tlbx.register("mate", tools.cxUniform, indpb = 0.5)
tlbx.register("mutate", self_adaptive_mutate, indpb=0.05)
tlbx.register("select",uniform_parent)
tlbx.register('survival',natural_selection)
tlbx.register("Doomsday",Doomsday)

OffProb = 0.8


fitns = list(map(tlbx.evaluate, Pop))


for ind, fit in zip(Pop, fitns):
    ind.fitness.values = fit

fit = [ind.fitness.values[0] for ind in Pop]
maxval = np.max(fit)
index = fit.index(maxval)
log.record(gen = n_gen, meanfit = np.mean(fit), varfit = np.var(fit), stdfit = np.std(fit), maxfit = maxval, optweightcombination = Pop[index])

while max(fit) < 100 and n_gen < max_gens:
    n_gen += 1
    print("---------------------Generation {}-------------------------".format(n_gen + 1))
    offspring = tlbx.select(Pop)
    offspring = list(map(tlbx.clone, offspring))
    
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < OffProb:
            tlbx.mate(child1,child2)
            del child1.fitness.values
            del child2.fitness.values
        
    sigma = modify_sigma(tau, sigma=sigma)

    for mutant in offspring:
        if random.random() < mut_prob:
            tlbx.mutate(mutant, sigma)
            del mutant.fitness.values

    new_ind = [ind for ind in offspring if not ind.fitness.valid]    
    fitns = list(map(tlbx.evaluate, new_ind))

    for ind, fit in zip(new_ind, fitns):
        ind.fitness.values = fit

    Pop[:] = tlbx.survival(offspring, len(Pop))

    fits = [ind.fitness.values[0] for ind in Pop]

    log.record(gen = n_gen, meanfit = np.mean(fits), varfit = np.var(fits), stdfit = np.std(fits), maxfit =  np.max(fits), optweightcombination = Pop[fits.index(np.max(fits))])


    if best.fitness.valid != True or best.fitness.values <= Pop[index].fitness.values:
        best = tlbx.clone(Pop[index])
    
    if log[n_gen].get("meanfit") <= log[n_gen - 1].get("meanfit") : 
        noimprovement += 1
    else :
        noimprovement = 0
    
    if noimprovement > 15:
        print('~~~~~~~~~~DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOM~~~~~~~~~~')
        del log[n_gen]
        tlbx.Doomsday(Pop, fits, sigma)
        fits = [ind.fitness.values[0] for ind in Pop]
        log.record(gen = n_gen, meanfit = np.mean(fits), varfit = np.var(fits), stdfit = np.std(fits), maxfit =  np.max(fits), optweightcombination = Pop[fits.index(np.max(fits))])


    
print(log.select("meanfit"))
print(best.fitness.values)

fig, pl = plt.subplots(2)

# plot with epidemic
pl[0].plot(log.select("gen"), log.select("meanfit"))
pl[1].plot(log.select("gen"), log.select("stdfit"))
pl[1].set_xlabel('Generations')
plt.show()


print(log.select("meanfit"))
print(best.fitness.values)
