'''
algorithmB.py ...
'''



# imports framework
import sys
import os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import glob, os
from deap import base, creator, tools
import random
import numpy as np

experiment_name = 'algorithmB'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# random.seed(1)

# enemy(ies) to play against
enemy = 2

if not os.path.exists(experiment_name + '/enemy {}'.format(enemy)):
    os.makedirs(experiment_name + '/enemy {}'.format(enemy))
# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  speed="fastest",
                  logs="off")

run_mode = 'train'

# standard variables
n_hidden = 10
pop_size = 50
n_gens = 20
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
upper_w = 1
lower_w = -1

# sigma for normal dist, tau constant,  mut_prob prop mutation for individu
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
env.state_to_log() # checks environment state

# register and create deap functions and classes
def register_deap_functions():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, lifepoints=1)

    tlbx.register("atrr_float", np.random.uniform, low=lower_w, high=upper_w)
    tlbx.register("individual", tools.initRepeat, creator.Individual, tlbx.atrr_float, n=n_weights)
    tlbx.register("population", tools.initRepeat, list, tlbx.individual, n=pop_size)
    tlbx.register("evaluate", evaluate)

    tlbx.register("blend_crossover", blend_crossover, alpha=.5)
    tlbx.register("cxUniform", tools.cxUniform , indpb=.05)

    tlbx.register("uniform_parent", uniform_parent)

    tlbx.register("mutate", self_adaptive_mutate, indpb=0.05)

    tlbx.register("select", uniform_parent)
    tlbx.register('survival', natural_selection)

# evaluate individual
def evaluate(individual):
    f,p,e,t = env.play(pcont=individual)
    individual.lifepoints = p
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

# blend crossover function, returns two children
def blend_crossover(parent1, parent2, alpha=0.5):
    d = abs(parent1 - parent2)

    child1 = creator.Individual(np.random.uniform(np.minimum(parent1, parent2)-alpha*d, np.maximum(parent1, parent2)+alpha*d))
    child2 = creator.Individual(np.random.uniform(np.minimum(parent1, parent2)-alpha*d, np.maximum(parent1, parent2)+alpha*d))
    return child1, child2

def uniform_parent(pop): # the pop the portion of total pop you want as chosen individuals

    """the selection for the 'mating population' is created by uniform distribution
    and is 3 times the size of the orginial population"""
    chosen_ind = []
    len_matingpop = len(pop)

    for ind in range(0, len_matingpop):
        num = random.randint(0, (len(pop)-1))
        chosen_ind.append(pop[num])

    return chosen_ind

# checks the an individual for its values, if over boundary
def check_bounds(ind, lower_w, upper_w):
    for i in range(len(ind)):
        if ind[i] > upper_w:
            ind[i] = upper_w
        elif ind[i] < lower_w:
            ind[i] = lower_w
    return ind

# replace worst half of pop by mutating every allele in genome
def doomsday(pop, pop_fit, sigma):
    worst = len(pop)//2
    order = np.argsort(pop_fit)
    orderasc = order[0:worst]

    for idx in orderasc:
        tlbx.mutate(pop[idx], sigma, indpb=1)
        new_fit = tlbx.evaluate(pop[idx])
        pop[idx].fitness.values = new_fit

    pop_fit = [ind.fitness.values[0] for ind in pop]
    return pop

# register deap functions
register_deap_functions()

for n_sim in range(10):

    if not os.path.exists(experiment_name+'/enemy {}/sim {}'.format(enemy, n_sim+1)):
        os.makedirs(experiment_name+'/enemy {}/sim {}'.format(enemy, n_sim+1))
    print("-------------------------Simulation {}----------------------------------------------".format(n_sim+1))
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

    file_aux  = open(experiment_name+'/enemy {}/sim {}/results.txt'.format(enemy, n_sim+1), 'a')
    print( '\n GENERATION '+str(0)+ ' Ave fit: '+str(round(mean,6))+ ' Std:  '+str(round(std,6))+ ' Best '+str(round(pop_fit[best],6)) + ' Ave life: ' + str(round(mean_life,6)))

    file_aux.write('GEN ' + 'Mean fit ' + 'Std ' + 'Best ' + 'Ave life' + '\n')
    file_aux.write(str(0)+' '+str(round(mean,6))+' '+str(round(std,6))+' '+str(round(pop_fit[best],6)) + ' ' + str(round(mean_life, 6)) +'\n')
    file_aux.close()

    for n_gen in range(n_gens):
        print("---------------------Generation {}-------------------------".format(n_gen + 1))

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



        # save result
        file_aux  = open(experiment_name+'/enemy {}/sim {}/results.txt'.format(enemy, n_sim+1), 'a')
        print( '\n GENERATION '+str(n_gen + 1)+' Ave fit: '+str(round(mean,6))+' Std:  '+str(round(std,6))+' Best '+str(round(pop_fit[best],6)) + ' Ave life: ' + str(round(mean_life,6)))
        file_aux.write(str(n_gen+1)+' '+str(round(mean,6))+' '+str(round(std,6))+' '+str(round(pop_fit[best],6)) +' ' + str(round(mean_life, 6)) +'\n')
        file_aux.close()

        if pop_fit[best] > best_overall:
            np.savetxt(experiment_name + '/enemy {}/sim {}/best_solution.txt'.format(enemy, n_sim+1), pop[best])
            noimprove = 0
        else:
            noimprove += 1

        if noimprove > n_gens//4:
            print("Doomsday")
            pop = doomsday(pop, pop_fit, sigma)
            noimprove = 0

    print("average of generations: ", average_pops)

    np.savetxt(experiment_name + "/enemy {}/sim {}/mean_gen.txt".format(enemy, n_sim+1), average_pops)
    np.savetxt(experiment_name + "/enemy {}/sim {}/std_gen.txt".format(enemy, n_sim+1), std_pops)
    np.savetxt(experiment_name + "/enemy {}/sim {}/best_per_gen.txt".format(enemy, n_sim+1), best_per_gen)
    np.savetxt(experiment_name + "/enemy {}/sim {}/mean_life.txt".format(enemy, n_sim+1), player_means)

    average_pop = []
    std_pops = []
    best_per_gen = []
    player_means = []
    best_overall = 0
    noimprove = 0
