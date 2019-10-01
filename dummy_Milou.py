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
max_gens = 2
n_gen = 0
noimprovement = 0

# sigma for normal dist, tao constant,  pm prop mutation for individu
sigma = 1
tao = 1./np.sqrt(n_weights)
pm = 1./n_pop


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
def modify_sigma(tao, sigma=sigma):
    return sigma * np.exp(tao*np.random.normal(0,1))

# mutates alles of gen with p indpb
def self_adaptive_mutate(individual, sigma, indpb):
    mu = 0
    print('prob allel:', indpb)
    print('newsig:',sigma)
    normal_dist = np.random.normal(mu, sigma, len(individual))
    xadd = np.where(np.random.random(normal_dist.shape) < 1-indpb, 0, normal_dist)
    return individual + xadd

def Doomsday(pop):
    return pop


    
    

tlbx.register("evaluate", EvaluateFit)
tlbx.register("mate", tools.cxUniform, indpb = 0.5)
tlbx.register("mutate", self_adaptive_mutate, indpb=(1/n_weights))
tlbx.register("select",tools.selTournament, tournsize = 3)
tlbx.register('survival',tools.selTournament, tournsize = 3 )
# tlbx.register("Doomsday",doomsday)

OffProb = 0.8
# tlbx.register("normialise", Normalise)

# evaluate initial pop

fitns = list(map(tlbx.evaluate, Pop))
# fitnsnorm = list(map(lambda x: tlbx.normialise(x, fitns), fitns))
# print(fitnsnorm)

for ind, fit in zip(Pop, fitns):
    ind.fitness.values = fit

fit = [ind.fitness.values[0] for ind in Pop]
maxval = np.max(fit)
index = fit.index(maxval)
log.record(gen = n_gen, meanfit = np.mean(fit), varfit = np.var(fit), stdfit = np.std(fit), maxfit = maxval, optweightcombination = Pop[index])

while max(fit) < 100 and n_gen < max_gens:
    n_gen += 1
    print("---------------------Generation %i-------------------------", n_gen)
    offspring = tlbx.select(Pop, len(Pop)*3)
    offspring = list(map(tlbx.clone, offspring))
    
    for child1, child2 in zip(offspring[::1], offspring[1::2]):
        if random.random() < OffProb:
            tlbx.mate(child1,child2)
            del child1.fitness.values
            del child2.fitness.values
        
    sigma = modify_sigma(tao, sigma=sigma)

    for mutant in offspring:
        if random.random() < pm:
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
    
    if noimprovement > 1:
        print('~~~~~~~~~~DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOM~~~~~~~~~~')
        del log[n_gen]
        Doomsday(Pop)
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
