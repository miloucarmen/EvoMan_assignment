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

log = tools.Logbook()
tlbx = base.Toolbox()
env.state_to_log() # checks environment state

def register_deap_functions():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    tlbx.register("atrr_float", np.random.uniform, low=lower_w, high=upper_w)
    tlbx.register("individual", tools.initRepeat, creator.Individual, tlbx.atrr_float, n=n_weights)
    tlbx.register("population", tools.initRepeat, list, tlbx.individual, n=pop_size)
    tlbx.register("evaluate", evaluate)

    tlbx.register("blend_crossover", blend_crossover, alpha=.5)
    tlbx.register("cxUniform", tools.cxUniform , indpb=.05)

    tlbx.register("uniform_parent", uniform_parent)

    tlbx.register("mutate", self_adaptive_mutate, indpb=0.05)

    tlbx.register("select",tools.selTournament, tournsize = 3)
    tlbx.register('survival', natural_selection)

# evaluate individual
def evaluate(x):
    f,p,e,t = env.play(pcont=x)
    return f,

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

    child1 = creator.Individual(np.random.uniform(parent1-alpha*d, parent1+alpha*d))
    child2 = creator.Individual(np.random.uniform(parent2-alpha*d, parent2+alpha*d))
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

register_deap_functions()

pop = tlbx.population()
fitns = [tlbx.evaluate(individual) for individual in pop]

average_pops.append(np.array(fitns).mean())


for ind, fit in zip(pop, fitns):
    ind.fitness.values = fit

for n_gen in range(n_gens):
    print("---------------------Generation {}-------------------------".format(n_gen + 1))

    offspring = tlbx.select(pop, pop_size)
    offspring = list(map(tlbx.clone, offspring))
    new_pop = []


    sigma = modify_sigma(tau, sigma=sigma)

    for parent1, parent2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < mate_prob:
            for i in range(4):
                child1, child2 = karina_crossover(parent1,parent2)
                new_pop.append(child1), new_pop.append(child2)
        else:
            new_pop.append(parent1), new_pop.append(parent2)

    print("new_popsize: ", len(new_pop))

    for mutant in new_pop:
        if random.random() < mut_prob:

            tlbx.mutate(mutant, sigma)

    new_pop = [check_bounds(ind, lower_w, upper_w) for ind in new_pop]
    new_pop_fitness = [tlbx.evaluate(ind) for ind in new_pop]

    for ind, fit in zip(new_pop, new_pop_fitness):
        ind.fitness.values = fit

    pop[:] = tlbx.survival(new_pop, pop_size)

    fits = np.array([ind.fitness.values for ind in pop])
    print("Fitnesses current population:", fits)
    best_curr_pop = pop[np.argmax(fits)]
    average_pop = fits.mean()

    average_pops.append(average_pop)

    if best_curr_pop.fitness.values[0] > best_ind[0]:
        best_ind[0] = best_curr_pop.fitness.values[0]
        best_ind[1] = best_curr_pop

print("-----------------------------------------------------------------------")
print("average of generations: ", average_pops)
print("Best solution: ", best_ind[0])

np.savetxt(experiment_name + "/solution_enemy{}.txt".format(enemy), best_ind[1])
np.savetxt(experiment_name + "/average_gens_enemy{}.txt".format(enemy), average_pops)
