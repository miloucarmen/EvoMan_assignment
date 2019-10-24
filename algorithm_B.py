###############################################################################
# Course: Evolutionary Computing                                              #
# Summary: With this file, a simulation of the game evoman can be run. The    #
# player will be controlled by a neural network. Neural networks can be       #
# trained or tested. This is the implementation of algorithm B                #
###############################################################################

# Import evoman framework
import sys
import os
from numba import jit, cuda
import numpy as np

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import glob, os
from deap import base, creator, tools
import random
import numpy as np

###############################################################################
################################# Setup #######################################
###############################################################################

enemy = [2,6]

experiment_name = 'algorithmB_generalist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing
# against static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2,6],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  speed="fastest",
                  logs="off")

run_mode = 'train'

# Standard variables
n_hidden = 10
pop_size = 50
n_gens = 20
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
upper_w = 1
lower_w = -1

sigma = 1
tau = 1/np.sqrt(n_weights)
mut_prob = 0.1
mate_prob = 0.8
average_pops = []
std_pops = []
best_per_gen = []
player_means = []

best_overall = 0
noimprove = 0

log = tools.Logbook()
tlbx = base.Toolbox()
env.state_to_log()

###############################################################################
################################ Functions ####################################
###############################################################################

# Register and create deap functions and classes
def register_deap_functions():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax,
                   lifepoints=1)

    tlbx.register("atrr_float", np.random.uniform, low=lower_w, high=upper_w)
    tlbx.register("individual", tools.initRepeat, creator.Individual,
                  tlbx.atrr_float, n=n_weights)
    tlbx.register("population", tools.initRepeat, list, tlbx.individual,
                  n=pop_size)
    tlbx.register("evaluate", evaluate)

    tlbx.register("blend_crossover", blend_crossover, alpha=.5)
    tlbx.register("cxUniform", tools.cxUniform , indpb=.05)

    tlbx.register("uniform_parent", uniform_parent)

    tlbx.register("mutate", self_adaptive_mutate, indpb=0.05)

    tlbx.register("select", uniform_parent)
    tlbx.register('survival', natural_selection)

# Evaluate individual
def evaluate(individual):
    f,p,e,t = env.play(pcont=individual)
    individual.lifepoints = p
    return f,

# Changes sigma over time
def modify_sigma(tau, sigma=sigma):
    return sigma * np.exp(tau*np.random.normal(0,1))

# Mutates alles of gen with p indpb
def self_adaptive_mutate(individual, sigma, indpb, mu = 0):
    normal_dist = np.random.normal(mu, sigma, len(individual))
    xadd = np.where(np.random.random(normal_dist.shape) < 1-indpb, 0,
                    normal_dist)
    return individual + xadd

# Natural selection of population, without replacement
def natural_selection(selectionpop, pop_size, n_select=3):
    fitselect = [ind.fitness.values[0] for ind in selectionpop]
    pop = []
    for _ in range(pop_size):
        if len(selectionpop) < 3:
            n_select = len(selectionpop)
        idx_inds = random.sample(range(len(fitselect)), n_select)
        fitness_inds = np.array(fitselect)[idx_inds]
        best_idx = idx_inds[np.argmax(fitness_inds)]
        pop.append(selectionpop.pop(best_idx))
        fitselect.pop(best_idx)
    return pop

# Blend crossover function, returns two children
def blend_crossover(parent1, parent2, alpha=0.5):
    d = abs(parent1 - parent2)

    child1 = creator.Individual(np.random.uniform(parent1-alpha*d, parent1+alpha*d))
    child2 = creator.Individual(np.random.uniform(parent2-alpha*d, parent2+alpha*d))
    return child1, child2

# The portion of the total population you want as chosen individuals.
def uniform_parent(pop):
    chosen_ind = []
    len_matingpop = len(pop)

    for ind in range(0, len_matingpop):
        num = random.randint(0, (len(pop)-1))
        chosen_ind.append(pop[num])

    return chosen_ind

# Check if the newly assigned weights don't go over or under the set
# boundaries.
def check_bounds(ind, lower_w, upper_w):
    for i in range(len(ind)):
        if ind[i] > upper_w:
            ind[i] = upper_w
        elif ind[i] < lower_w:
            ind[i] = lower_w
    return ind

register_deap_functions()

###############################################################################
############################# Evolution #######################################
###############################################################################

for n_sim in range(10):

    if not os.path.exists(experiment_name+'/sim {}'.format(n_sim+1)):
        os.makedirs(experiment_name+'/sim {}'.format(n_sim+1))
    print("-------------Simulation {}-------------------".format(n_sim+1))
    # initializes population at random
    pop = tlbx.population()
    pop_fit = [tlbx.evaluate(ind) for ind in pop]

    for ind, fit in zip(pop, pop_fit):
        ind.fitness.values = fit

    pop_fit = [ind.fitness.values[0] for ind in pop]
    player_life = [ind.lifepoints for ind in pop]

    best = np.argmax(pop_fit)
    std = np.std(pop_fit)
    mean = np.mean(pop_fit)
    mean_life = np.mean(player_life)

    player_means.append(mean_life)
    average_pops.append(mean)
    std_pops.append(std)
    best_per_gen.append(pop_fit[best])

    file_aux  = open(experiment_name+'/sim {}/results.txt'.format(n_sim+1), 'a')
    print( '\n GENERATION '+str(0)+ ' Ave fit: '+str(round(mean,6))+ ' Std:  '+str(round(std,6))+ ' Best '+str(round(pop_fit[best],6)) + ' Ave life: ' + str(round(mean_life,6)))

    file_aux.write('GEN ' + 'Mean fit ' + 'Std ' + 'Best ' + 'Ave life' + '\n')
    file_aux.write(str(0)+' '+str(round(mean,6))+' '+str(round(std,6))+' '+str(round(pop_fit[best],6)) + ' ' + str(round(mean_life, 6)) +'\n')
    file_aux.close()

    for n_gen in range(n_gens):
        print("------------Generation {}-------------".format(n_gen + 1))

        offspring = tlbx.select(pop)
        offspring = list(map(tlbx.clone, offspring))
        new_pop = []

        sigma = modify_sigma(tau, sigma=sigma)

        for parent1, parent2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < mate_prob:
                for i in range(4):
                    child1, child2 = tlbx.blend_crossover(parent1,parent2)
                    new_pop.append(child1), new_pop.append(child2)
            else:
                new_pop.append(parent1), new_pop.append(parent2)

        for mutant in new_pop:
            if random.random() < mut_prob:
                tlbx.mutate(mutant, sigma)

        print('Length of all offspring: ', len(new_pop))
        new_pop = [check_bounds(ind, lower_w, upper_w) for ind in new_pop]
        new_pop_fitness = [tlbx.evaluate(ind) for ind in new_pop]

        for ind, fit in zip(new_pop, new_pop_fitness):
            ind.fitness.values = fit

        pop[:] = tlbx.survival(new_pop, pop_size)
        pop_fit = [ind.fitness.values[0] for ind in pop]
        player_life = [ind.lifepoints for ind in pop]


        best = np.argmax(pop_fit)
        std = np.std(pop_fit)
        mean = np.mean(pop_fit)
        mean_life = np.mean(player_life)

        player_means.append(mean_life)
        average_pops.append(mean)
        std_pops.append(std)
        best_per_gen.append(pop_fit[best])

        print("Pop:", pop_fit)

###############################################################################
############################# Results #########################################
###############################################################################

        file_aux  = open(experiment_name+'/sim {}/results.txt'.format(n_sim+1), 'a')
        print( '\n GENERATION '+str(n_gen + 1)+' Ave fit: '+str(round(mean,6))+' Std:  '+str(round(std,6))+' Best '+str(round(pop_fit[best],6)) + ' Ave life: ' + str(round(mean_life,6)))
        file_aux.write(str(n_gen+1)+' '+str(round(mean,6))+' '+str(round(std,6))+' '+str(round(pop_fit[best],6)) +' ' + str(round(mean_life, 6)) +'\n')
        file_aux.close()

        if pop_fit[best] > best_overall:
            best_overall = pop_fit[best]
            np.savetxt(experiment_name + '/sim {}/best_solution.txt'.format(n_sim+1), pop[best])


    print("average of generations: ", average_pops)

    np.savetxt(experiment_name + "/sim {}/mean_gen.txt".format(n_sim+1), average_pops)
    np.savetxt(experiment_name + "/sim {}/std_gen.txt".format(n_sim+1), std_pops)
    np.savetxt(experiment_name + "/sim {}/best_per_gen.txt".format(n_sim+1), best_per_gen)
    np.savetxt(experiment_name + "/sim {}/mean_life.txt".format(n_sim+1), player_means)

    average_pop = []
    std_pops = []
    best_per_gen = []
    player_means = []
    best_overall = 0
