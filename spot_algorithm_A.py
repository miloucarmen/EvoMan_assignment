###############################################################################
# Course: Evolutionary Computing                                              #  
# Summary: With this file, a simulation of the game evoman can be run. The    #
# player will be controlled by a neural network. Neural networks can be       # 
# trained or tested. This is the implementation of algorithm A                #
###############################################################################

# Import evoman framework
import sys
import os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import glob
from deap import base
from deap import creator
from deap import tools
import random
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
################################# Setup #######################################
###############################################################################

# import arguments given by the user
arguments = sys.argv[1]
spot_parameters = eval(arguments.split()[0])

run_mode = 'train'
experiment_name = 'spot_algorithm_A'
# enemies = [1,2,3]
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize environment with an ai player using a random controller, playing
# against a static enemy
env = Environment(experiment_name = experiment_name,
                  enemies = [1,2,3],
                  multiplemode="yes",
                  playermode = 'ai',
                  player_controller = player_controller(),
                  enemymode = 'static',
                  speed = 'fastest')


# Standard variables
n_train_simulations = 10
n_hidden = 10
n_pop = 50
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 
max_gens = 20
noimprovement = 0
low_bound = -1
upper_bound = 1
average_pops = []
std_pops = []
best_per_gen = []
player_means = []


sigma = 1
tau = 1/np.sqrt(n_weights)
mut_prob = spot_parameters[1]
OffProb = spot_parameters[0]

# Create a deap base object, containing the fitness function and the individual
# type.
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', np.ndarray, fitness = creator.FitnessMax, lifepoint = 1)

tlbx = base.Toolbox()

# Register exact definition of an individual: An n_weights-long array of random
# numbers.
tlbx.register('atrr_float', random.uniform, low_bound, upper_bound)
tlbx.register('individual', tools.initRepeat, creator.Individual, tlbx.atrr_float, n = n_weights)

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
def Doomsday(pop, fit):
    worst = int(n_pop/4) 
    order = np.argsort(fit)
    orderasc = order[0:worst]
    
    for i in orderasc:
        self_adaptive_mutate(pop[i], 1, indpb=1)
        newfit = tlbx.evaluate(pop[i])
        pop[i].fitness.values = newfit

    return pop

# The pop the portion of total pop you want as chosen individuals
def uniform_parent(pop):
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
# if run_mode =='test':

#             df([enemy][2*i] = tlbx.evaluate(bsol_bart)
#             df[enemy][(2*i + 1)] = = tlbx.evaluate(bsol_Milou)
#             df['Algorithm'][2*i] = 1
#             df['Algorithm'][(2*i + 1)] = 2



#     ax = sns.boxplot(x='enemy', y="Fitness", hue="smoker", data=[f_bart, f_Milou], palette="Set3")


    # sys.exit(0)

# Run train mode
if run_mode == 'train':

    Logs = {}

    if not os.path.exists(experiment_name+'/generalist'):
        os.makedirs(experiment_name+'/generalist')

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
    
    bestmean = np.mean(fit)
    maxgen = np.max(fit)
    std = np.std(fit)
    mean_life = np.mean(lifepoint)

    player_means.append(mean_life)
    average_pops.append(bestmean)
    std_pops.append(std)
    best_per_gen.append(maxgen)


    # Log results
    log.record(gen = n_gen, meanfit = np.mean(fit), varfit = np.var(fit), 
                stdfit = np.std(fit), maxfit =  np.max(fit), 
                avelifepoint = np.mean(lifepoint))

    # Save stats of first generation
    
    file_aux  = open(experiment_name+'/generalist/results' +'.txt','a')
    print( '\n GENERATION '+str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
    
    file_aux.write('\n' +'GEN ' + 'Mean fit ' + 'Std ' + 'Best ' + 'Ave life' + '\n')
    file_aux.write(str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
    file_aux.close()

    # Get best instance of first generation
    index = fit.index(np.max(fit))
    ind = Pop[index]
    best = tlbx.clone(ind)

    # Save file with the best solution
    np.savetxt(experiment_name+ '/generalist/best' + '.txt', best)

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

        maxgen = np.max(fits)
        std = np.std(fits)
        mean = np.mean(fits)
        mean_life = np.mean(lifepoint)

        # Log results
        log.record(gen = n_gen, meanfit = mean, 
                    varfit = np.var(fits), stdfit = std, 
                    maxfit = np.max(fits), 
                    avelifepoint = mean_life)


        # Checks if the best instance of the current generation is the 
        # overall best so far.
        if best.fitness.values < log[n_gen].get('maxfit'):

            index = fit.index(np.max(fit))
            ind = Pop[index]
            best = tlbx.clone(ind)

            np.savetxt(experiment_name+'/generalist/best' + '.txt', best)

        # Check if meanfit keeps improving
        if log[n_gen].get('meanfit') <= bestmean : 
            noimprovement += 1
        else :
            noimprovement = 0
            bestmean = np.mean(fits)
        
        # If the period without improvement is to long, doomsday will be 
        # called upon the generation.
        if noimprovement > (max_gens//10):

            # deletes log domed gen, initialize a new population and fitness
            del log[n_gen]

            tlbx.Doomsday(Pop, fits)
            fits = [ind.fitness.values[0] for ind in Pop]
            lifepoint = [ind.lifepoint for ind in Pop]
            noimprovement = 0

            # Log results
            log.record(gen = n_gen, meanfit = np.mean(fits), 
                    varfit = np.var(fits), stdfit = np.std(fits), 
                    maxfit = np.max(fits), 
                    avelifepoint = np.mean(lifepoint))

        # Save results in text files
        file_aux  = open(experiment_name+'/generalist/results' +'.txt','a')
        print('\n GENERATION '+str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
        file_aux.write('\n'+ str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
        file_aux.close()

        print(np.mean(fits))
        maxgen = np.max(fits)
        std = np.std(fits)
        mean = np.mean(fits)
        mean_life = np.mean(lifepoint)

        player_means.append(mean_life)
        average_pops.append(bestmean)
        std_pops.append(std)
        best_per_gen.append(maxgen)

        np.savetxt(experiment_name+ '/generalist/mean_gen.txt', average_pops)
        np.savetxt(experiment_name+ '/generalist/std_gen.txt', std_pops)
        np.savetxt(experiment_name+ '/generalist/best_per_gen.txt', best_per_gen)
        np.savetxt(experiment_name+ '/generalist/mean_life.txt', player_means)
        


        Logs = log
    