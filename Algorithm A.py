###############################################################################
# Course: Evolutionary Computing                                              #  
# Summary: With this file, a simulation of the game evoman can be run. The    #
# player will be controlled by a neural network. Neural networks can be       # 
# trained or tested.                                                          #
###############################################################################

# Import evoman framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import glob, os
from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt

run_mode = 'train'
experiment_name = 'offi_Milou'
enemy = 2 
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

###############################################################################
################################# Setup #######################################
###############################################################################

# Initialize environment with an ai player using a random controller, playing
# against a static enemy
env = Environment(experiment_name = experiment_name,
                  enemies = [enemy],
                  playermode = 'ai',
                  player_controller = player_controller(),
                  enemymode = 'static',
                  speed = 'fastest')

# Global variables
n_train_simulations = 3
n_hidden = 10
n_pop = 2
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 
max_gens = 2
noimprovement = 0
low_bound = -1
upper_bound = 1

sigma = 1
tau = 1/np.sqrt(n_weights)
mut_prob = 0.1
OffProb = 0.8

# Create a deap base object, containing the fitness function and the individual
# type.
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', np.ndarray, fitness = creator.FitnessMax, 
			    lifepoint = 1)

tlbx = base.Toolbox()

# Register exact definition of an individual: An n_weights-long array of random
# numbers.
tlbx.register('atrr_float', random.uniform, low_bound, upper_bound)
tlbx.register('individual', tools.initRepeat, creator.Individual, 
              tlbx.atrr_float, n = n_weights)

# Population is n_pop long list of individuals
tlbx.register('Population', tools.initRepeat, list, tlbx.individual, n = n_pop)

###############################################################################
################################ Functions ####################################
###############################################################################

# Evaluation
def EvaluateFit(individual):
    f,p,e,t = env.play(pcont=individual)
    individual.lifepoint = p
    return f,

# Change sigma over time
def modify_sigma(tau, sigma=sigma):
    return sigma * np.exp(tau*np.random.normal(0,1))

# mutates alles of gen with p indpb
def self_adaptive_mutate(individual, sigma, indpb):
    mu = 0
    normal_dist = np.random.normal(mu, sigma, len(individual))
    xadd = np.where(np.random.random(normal_dist.shape) < 1-indpb, 0,
                    normal_dist)
    individual = individual + xadd
    for i in range(len(individual)):
        if individual[i] > upper_bound:
            individual[i] = upper_bound
        elif individual[i] < low_bound:
            individual[i] = low_bound
    return individual

# natural selection of population, without chance of picking the same 
# individual twice
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

# Replaces 25% of population
def Doomsday(pop, fit, sigma):
    worst = int(n_pop/4) 
    order = np.argsort(fit)
    orderasc = order[0:worst]

    for i in orderasc:
        self_adaptive_mutate(pop[i], sigma, indpb=1)
        newfit = tlbx.evaluate(pop[i])
        pop[i].fitness.values = newfit

    return pop

# The pop the portion of total pop you want as chosen individuals
def uniform_parent(pop):

    '''the selection for the 'mating population' is created by uniform distribution
    and is 3 times the size of the orginial population'''
    chosen_ind = []
    len_matingpop = 3 * len(pop)

    for ind in range(0, len_matingpop):
        num = random.randint(0, (len(pop)-1))
        chosen_ind.append(pop[num])

    return chosen_ind

# registers all functions to toolbox
tlbx.register('evaluate', EvaluateFit)
tlbx.register('mate', tools.cxBlend)
tlbx.register('mutate', self_adaptive_mutate, indpb=0.05)
tlbx.register('select', uniform_parent)
tlbx.register('survival', tools.selTournament, tournsize = 3)
tlbx.register('Doomsday', Doomsday)

###############################################################################
############################# Evolution #######################################
###############################################################################

# Run test mode
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print(bsol)
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    tlbx.evaluate(bsol)

    sys.exit(0)

# Run train mode
if run_mode == 'train':

    Logs = {}
    for i in range(n_train_simulations):
        print('new')
        n_gen = 0

        log = tools.Logbook()
        Pop = tlbx.Population()
        best = tlbx.individual()

        fitns = list(map(tlbx.evaluate, Pop))
        print(Pop[0].lifepoint)

        for ind, fit in zip(Pop, fitns):
            ind.fitness.values = fit

        fit = [ind.fitness.values[0] for ind in Pop]
        maxval = np.max(fit)
        index = fit.index(maxval)
        lifepoint = [ind.lifepoint for ind in Pop]

        # Log results
        log.record(gen = n_gen, meanfit = np.mean(fit), varfit = np.var(fit), 
                   stdfit = np.std(fit), maxfit =  np.max(fit), 
                   avelifepoint = np.mean(lifepoint))

        # Save stats of first generation
        file_aux  = open(experiment_name+'/results'+ str(i) +'.txt','a')
        file_aux.write('\n\ngen mean std max')
        print( '\n GENERATION '+str(n_gen)+' '
               +str(round(log[n_gen].get('meanfit'),6))+' '
               +str(round(log[n_gen].get('stdfit'),6))+' '
               +str(round(log[n_gen].get('maxfit'),6)))
        file_aux.write('\n'+ str(n_gen)+' '
                       +str(round(log[n_gen].get('meanfit'),6))+' '
                       +str(round(log[n_gen].get('stdfit'),6))+' '
                       +str(round(log[n_gen].get('maxfit'),6)))
        file_aux.close()

        # Get best instance of first generation
        index = fit.index(np.max(fit))
        ind = Pop[index]
        best = tlbx.clone(ind)

        # Save file with the best solution
        np.savetxt(experiment_name+'/best'+ str(i) +'.txt', best)

        # Evolution
        for n_gen in range(max_gens):
            n_gen += 1
            print('------------Generation {}-------------'.format(n_gen))
            
            # Parent selection
            offspring = tlbx.select(Pop)
            offspring = list(map(tlbx.clone, offspring))
            
            # Select two parets from the offspring
            for parent1, parent2 in zip(offspring[::2], offspring[1::2]):
                
                # Mutate first parent
                if random.random() < mut_prob:  
                    tlbx.mutate(parent1, sigma)
                    del parent1.fitness.values

                # Mutate second parent
                if random.random() < mut_prob:  
                    tlbx.mutate(parent2, sigma)
                    del parent2.fitness.values

                # Mating
                if random.random() < OffProb:
                    tlbx.mate(parent1, parent2, random.random())
                    del parent1.fitness.values
                    del parent2.fitness.values

            # Evaluate new individuals in the population
            new_ind = [ind for ind in offspring if not ind.fitness.valid] 
            fitns = list(map(tlbx.evaluate, new_ind))

            for ind, fit in zip(new_ind, fitns):
                ind.fitness.values = fit

            # Replace old population and update all fitnesses values
            Pop[:] = tlbx.survival(offspring, len(Pop))
            fits = [ind.fitness.values[0] for ind in Pop]
            lifepoint = [ind.lifepoint for ind in Pop]

            # Log results
            log.record(gen = n_gen, meanfit = np.mean(fits), 
                       varfit = np.var(fits), stdfit = np.std(fits), 
                       maxfit = np.max(fits), 
                       avelifepoint = np.mean(lifepoint))
            
            # Save results in text files
            file_aux  = open(experiment_name+'/results'+ str(i) +'.txt','a')
            print('\n GENERATION '+str(n_gen)+' '
                  +str(round(log[n_gen].get('meanfit'),6))+' '
                  +str(round(log[n_gen].get('stdfit'),6))+' '
                  +str(round(log[n_gen].get('maxfit'),6)))
            file_aux.write('\n'+ str(n_gen)+' '
                           +str(round(log[n_gen].get('meanfit'),6))+' '
                           +str(round(log[n_gen].get('stdfit'),6))+' '
                           +str(round(log[n_gen].get('maxfit'),6)))
            file_aux.close()
            
            # Checks if the best instance of the current generation is the 
            # overall best so far.
            if best.fitness.values < log[n_gen].get('maxfit'):

                index = fit.index(np.max(fit))
                ind = Pop[index]
                best = tlbx.clone(ind)

                np.savetxt(experiment_name+'/best'+ str(i) +'.txt', best)

            # Check if meanfit keeps improving
            if log[n_gen].get('meanfit') <= log[n_gen - 1].get('meanfit') : 
                noimprovement += 1
            else :
                noimprovement = 0
            
            # If the period without improvement is to long, doomsday will be 
            # called upon the generation.
            if noimprovement > (max_gens//10):

                # Delete log of the doomed generation and initialize a new 
                # population and fitness
                del log[n_gen]
                tlbx.Doomsday(Pop, fits, sigma)
                fits = [ind.fitness.values[0] for ind in Pop]
                lifepoint = [ind.lifepoint for ind in Pop]
                noimprovement = 0

                # Log result
                log.record(gen = n_gen, meanfit = np.mean(fits), 
                           varfit = np.var(fits), stdfit = np.std(fits), 
                           maxfit =  np.max(fits), 
                           avelifepoint = np.mean(lifepoint))

                # Save result
                file_aux  = open(experiment_name+'/results'+ str(i) 
                	             +'.txt','a')
                file_aux.write('\n'+ str(n_gen)+' '
                                +str(round(log[n_gen].get('meanfit'),6))+' '\
                                +str(round(log[n_gen].get('stdfit'),6))+' '
                                +str(round(log[n_gen].get('maxfit'),6)))
                file_aux.close()

            Logs[i] = log
       
###############################################################################
############################# Results #########################################
###############################################################################

# Collect results
mean_average = []
std_average = []
lifepoint_average = []

for i in range(n_train_simulations):
    mean_average.append(Logs[i].select('meanfit'))
    std_average.append(Logs[i].select('stdfit'))
    lifepoint_average.append(Logs[i].select('avelifepoint'))

mean_average = np.sum(mean_average, 0)
std_average = np.sum(std_average, 0)
lifepoint_average = np.sum(lifepoint_average, 0)

mean_average[:] = [x / n_train_simulations for x in mean_average]
std_average[:] = [x / n_train_simulations for x in std_average]
lifepoint_average[:] = [x / n_train_simulations for x in lifepoint_average]
 
# Plot results   
fig, pl = plt.subplots(3)

pl[0].plot(Logs[0].select('gen'), mean_average)
pl[0].set_ylabel('Average fit')
pl[1].plot(Logs[0].select('gen'), std_average)
pl[1].set_ylabel('Average std')
pl[2].plot(Logs[0].select('gen'), lifepoint_average)
pl[2].set_ylabel('Average life')
pl[2].set_xlabel('Generations')
fig.savefig('enemy2.png')

plt.show()

print(log.select('meanfit'))
print(best.fitness.values)
