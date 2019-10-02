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

experiment_name = 'bart'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

random.seed(1)
enemy = 2

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
				  enemies=[enemy])

# standard variables
n_hidden = 10
pop_size = 5
n_gens = 2
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
upper_w = 1
lower_w = -1

# sigma for normal dist, tau constant,  mut_prob prop mutation for individu
sigma = 1
tau = 1/np.sqrt(n_weights)
mut_prob = 1/pop_size
mate_prob = 0.8
average_pops = []
best_ind = [0, []]

# evaluate individual
def simulate(ind):
    f,p,e,t = env.play(pcont=ind)
    return f

# evaluation of population
def evaluate(pop):
    return np.array([simulate(ind) for ind in pop])

# changes sigma over time
def modify_sigma(tau, sigma=sigma):
    return sigma * np.exp(tau*np.random.normal(0,1))

# mutates alles of gen with p indpb
def self_adaptive_mutate(individual, sigma, indpb, mu = 0):
    normal_dist = np.random.normal(mu, sigma, len(individual))
    xadd = np.where(np.random.random(normal_dist.shape) < 1-indpb, 0, normal_dist)
    return individual + xadd

# natural selection of population, without replacement
def natural_selection(selectionpop, pop_size):
    fitselect = [ind.fitness.values[0] for ind in selectionpop]
    pop = []
    for _ in range(pop_size):
        idx_inds = random.sample(range(len(fitselect)), 2)
        fitness_inds = np.array(fitselect)[idx_inds]
        best_idx = idx_inds[np.argmax(fitness_inds)]
        pop.append(selectionpop.pop(best_idx))
        fitselect.pop(best_idx)
    return pop

def karina_crossover(parent1, parent2):
    cross_prop = np.random.uniform(0,1)
    child1 = parent1*cross_prop + parent2*(1-cross_prop)
    child2 = parent1*(1-cross_prop) + parent2*cross_prop

    return creator.Individual(child1), creator.Individual(child2)

def blend_crossover(parent1, parent2, alpha=0.5):
    d =abs(parent1 - parent2)

    child1 = np.random.uniform(parent1-alpha*d, parent1+alpha*d)
    child2 = np.random.uniform(parent2-alpha*d, parent2+alpha*d)
    return child1, child2

def uniform_parent(pop): # the pop the portion of total pop you want as chosen individuals

    """the selection for the 'mating population' is created by uniform distribution
    and is 3 times the size of the orginial population"""
    chosen_ind = []
    len_matingpop = 3 * len(pop)

    for ind in range(0, len_matingpop):
        num = random.randint(0, (len(pop)-1))
        chosen_ind.append(pop[num])

    return chosen_ind

def check_bounds(ind, lower_w, upper_w):
    for i in range(len(ind)):
        if ind[i] > upper_w:
            ind[i] = upper_w
        elif ind[i] < lower_w:
            ind[i] = lower_w
    return ind


def main():
    pop = np.random.uniform(0,1, (pop_size, n_weights))
    print(np.shape(pop))
    fit_pop = evaluate(pop)
    print(fit_pop)


main()
