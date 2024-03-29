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
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

###############################################################################
################################# Setup #######################################
###############################################################################

run_mode = 'test'
experiment_name = 'algorithm_A'
enemy = []
if run_mode == 'test':
    enemy = [1]
    multiple = 'no'
else:
    enemy = [2,6]
    multiple = 'yes'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize environment with an ai player using a random controller, playing
# against a static enemy
env = Environment(experiment_name = experiment_name,
                  enemies = enemy,
                  multiplemode= multiple,
                  playermode = 'ai',
                  player_controller = player_controller(),
                  enemymode = 'static',
                  speed = 'fastest')


# Standard variables
n_train_simulations = 10
n_hidden = 10
n_pop = 100
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 
max_gens = 50
noimprovement = 0
low_bound = -1
upper_bound = 1
average_pops = []
std_pops = []
best_per_gen = []
player_means = []



sigma = 1
tau = 1/np.sqrt(n_weights)
mut_prob = 0.1
OffProb = 0.8

# Create a deap base object, containing the fitness function and the individual
# type.
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', np.ndarray, fitness = creator.FitnessMax, lifepoint = 1, enemylife = 1)

tlbx = base.Toolbox()

# Register exact definition of an individual: An n_weights-long array of random
# numbers.
tlbx.register('atrr_float', random.uniform, low_bound, upper_bound)
tlbx.register('individual', tools.initRepeat, creator.Individual, tlbx.atrr_float, n = n_weights)

# Population is n_pop long list of individuals
tlbx.register('Population', tools.initRepeat, list, tlbx.individual, n = n_pop)

bestboth = tlbx.individual()
###############################################################################
################################ Functions ####################################
###############################################################################

# Evaluation
def EvaluateFit(individual):
    f,p,e,t = env.play(pcont=individual)
    if run_mode == 'test':
        bestboth.lifepoint = p
        bestboth.enemylife = e
    else: 
        individual.lifepoint = p
        individual.enemylife = e
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

# # Replaces 25% of population
# def Doomsday(pop, fit):
#     worst = int(n_pop/4) 
#     order = np.argsort(fit)
#     orderasc = order[0:worst]
    
#     for i in orderasc:
#         self_adaptive_mutate(pop[i], 1, indpb=1)
#         newfit = tlbx.evaluate(pop[i])
#         pop[i].fitness.values = newfit

#     return pop

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


if run_mode == 'test':
    gain = pd.DataFrame(columns= (['Gain','Algorithm']))
    gainA = np.zeros(10)
    gainB = np.zeros(10)
    bestA = []
    bestB = []
    meanA = 0
    meanB = 0
    stdA = 0
    stdB = 0

    evaluationA= np.zeros((n_train_simulations, 8))
    evaluationB= np.zeros((n_train_simulations, 8))
    lifeA = np.zeros((n_train_simulations, 8))
    lifeB = np.zeros((n_train_simulations, 8))
    enemyA = np.zeros((n_train_simulations, 8))
    enemyB = np.zeros((n_train_simulations, 8))

    for i in range(n_train_simulations):
        bestA = np.loadtxt(experiment_name+'/generalist/sim {}'.format(i+1) + '/best.txt')
        bestB = np.loadtxt('algorithmB_generalist/sim {}/best_solution.txt'.format(i+1))
        gain = gain.append({'Gain' : 0,'Algorithm':'A'},ignore_index=True)
        gain = gain.append({'Gain' : 0, 'Algorithm':'B'},ignore_index=True)
        
        for j in range(8): 
            
            env.update_parameter('enemies',enemy) 

            evaluationA[i][j] = tlbx.evaluate(bestA)[0]
            lifeA[i][j] = bestboth.lifepoint 
            enemyA[i][j] = bestboth.enemylife
            gain['Gain'][2*i] = gain['Gain'][2*i]+ bestboth.lifepoint - bestboth.enemylife
            gainA[i] = gainA[i] + bestboth.lifepoint - bestboth.enemylife

            evaluationB[i][j] = tlbx.evaluate(bestB)[0]
            lifeB[i][j] = bestboth.lifepoint
            enemyB[i][j] = bestboth.enemylife
            gain['Gain'] [2*i + 1] = gain['Gain'] [2*i] + bestboth.lifepoint - bestboth.enemylife
            gainB[i] = gainB[i] + bestboth.lifepoint - bestboth.enemylife

            enemy[0] = enemy[0] + 1 

        enemy[0] = 1

    data = [gainA, gainB]

    plt.boxplot(data, labels=['A', 'B'])
    plt.title("Gain boxplot for both algorithms")

    plt.show()
    
    meanA = np.mean(gainA)
    meanB = np.mean(gainB)
    stdA = np.std(gainA)
    stdB = np.std(gainB)
    print(meanA, meanB, stdA, stdB)
    
    

    sys.exit(0)
###############################################################################
############################# Evolution #######################################
###############################################################################


# Run train mode
if run_mode == 'train':

    Logs = {}
    for i in range(n_train_simulations):
        if not os.path.exists(experiment_name+'/generalist/sim {}'.format(i+1)):
            os.makedirs(experiment_name+'/generalist/sim {}'.format(i+1))

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
        
        file_aux  = open(experiment_name+'/generalist/sim {}'.format(i+1)+'/results' +'.txt','a')
        print( '\n GENERATION '+str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
        
        file_aux.write('\n' +'GEN ' + 'Mean fit ' + 'Std ' + 'Best ' + 'Ave life' + '\n')
        file_aux.write(str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
        file_aux.close()

        # Get best instance of first generation
        index = fit.index(np.max(fit))
        ind = Pop[index]
        best = tlbx.clone(ind)

        # Save file with the best solution
        np.savetxt(experiment_name+ '/generalist/sim {}'.format(i+1) + '/best' + '.txt', best)

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
                    tlbx.mate(parent1, parent2, 0.5)
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

                np.savetxt(experiment_name+'/generalist/sim {}'.format(i+1)+'/best' + '.txt', best)

            # Save results in text files
            file_aux  = open(experiment_name+'/generalist/sim {}'.format(i+1)+'/results' +'.txt','a')
            print('\n GENERATION '+str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
            file_aux.write('\n'+ str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
            file_aux.close()

            print(np.mean(fits))
            maxgen = np.max(fits)
            std = np.std(fits)
            mean = np.mean(fits)
            mean_life = np.mean(lifepoint)

            player_means.append(mean_life)
            average_pops.append(mean)
            std_pops.append(std)
            best_per_gen.append(maxgen)

            np.savetxt(experiment_name+ '/generalist/sim {}'.format(i+1) + '/mean_gen.txt', average_pops)
            np.savetxt(experiment_name+ '/generalist/sim {}'.format(i+1) + '/std_gen.txt', std_pops)
            np.savetxt(experiment_name+ '/generalist/sim {}'.format(i+1) + '/best_per_gen.txt', best_per_gen)
            np.savetxt(experiment_name+ '/generalist/sim {}'.format(i+1) + '/mean_life.txt', player_means)



            Logs[i] = log
        average_pops = []
        std_pops = []
        best_per_gen = []
        player_means = []
       

