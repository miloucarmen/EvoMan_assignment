################################
#                              #
################################
# import evoman framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# import DEAP evolutionary algorithm framework and other necessary imports
import glob, os
from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt



# set seed, which mode and name experiment
run_mode = 'train'
experiment_name = 'offi_Milou'
enemies = [1,2,3]
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode='ai',
                  player_controller=player_controller(),
                  enemymode='static',
                  #level=2,
                  speed='fastest')

#global vars
n_train_simulations = 10
n_hidden = 10
n_pop = 50
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 
max_gens = 20
noimprovement = 0
low_bound = -1
upper_bound = 1

# sigma for normal dist, tao constant,  pm prop mutation for individu
sigma = 1
tau = 1/np.sqrt(n_weights)
mut_prob = 0.1
OffProb = 0.8

# create base for evo algorithm, fitness to be optimized and individual is an array and a fitness
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', np.ndarray, fitness = creator.FitnessMax, lifepoint = 1)

# create toolbox
tlbx = base.Toolbox()

# register exact def of individual it class Individual with a n_weights long initialized array of random numbers
tlbx.register('atrr_float', random.uniform, low_bound, upper_bound)
tlbx.register('individual', tools.initRepeat, creator.Individual, tlbx.atrr_float, n = n_weights)

# Population is n_pop long list of individuals
tlbx.register('Population', tools.initRepeat, list, tlbx.individual, n = n_pop)

################################ All functions ####################################33
# evaluation
def EvaluateFit(individual):
    f,p,e,t = env.play(pcont=individual)
    individual.lifepoint = p
    return f,

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

# natural selection of population, without chance of picking the same individual twice
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
        self_adaptive_mutate(pop[i], sigma, indpb=1)
        newfit = tlbx.evaluate(pop[i])
        pop[i].fitness.values = newfit

    return pop

def uniform_parent(pop): # the pop the portion of total pop you want as chosen individuals

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
###################################### End functions ########################################

# if test mode
# if run_mode =='test':
#     df = pd.DataFrame(index= [np.arange(n_train_simulations*2)], columns= (enemies + ['Algorithm']))

#     for i in range(10):
#         df[enemy][2*i] = 10 + i
#         df[enemy][(2*i + 1)] = 11 + i
        
#         print(df)
    
#     df = pd.DataFrame(columns = ['Algorithm'])
#     for enemy in enemies:

#         df['enemy' + str(enemy)] = 0

#         for i in range(n_train_simulations):
#             bsol_bart = np.loadtxt('dummy_bart/enemy'+ str(enemy) +'/sim '+ str(i) +'best_solution.txt')
#             bsol_Milou = np.loadtxt(experiment_name+'/best'+ str(i) + '.txt')

#             df([enemy][2*i] = tlbx.evaluate(bsol_bart)
#             df[enemy][(2*i + 1)] = = tlbx.evaluate(bsol_Milou)
#             df['Algorithm'][2*i] = 1
#             df['Algorithm'][(2*i + 1)] = 2



#     ax = sns.boxplot(x='enemy', y="Fitness", hue="smoker", data=[f_bart, f_Milou], palette="Set3")


    # sys.exit(0)


# train mode
if run_mode == 'train':

    Logs = {}
    for i in range(n_train_simulations):
        print('new')
        n_gen = 0
        # start logbook, create initial population and a individual for the future best solution.
        log = tools.Logbook()
        Pop = tlbx.Population()
        best = tlbx.individual()

        fitns = list(map(tlbx.evaluate, Pop))
        print(Pop[0].lifepoint)


        for ind, fit in zip(Pop, fitns):
            ind.fitness.values = fit

        # 
        fit = [ind.fitness.values[0] for ind in Pop]
        maxval = np.max(fit)
        index = fit.index(maxval)
        lifepoint = [ind.lifepoint for ind in Pop]
        
        # logs results
        log.record(gen = n_gen, meanfit = np.mean(fit), varfit = np.var(fit), stdfit = np.std(fit), maxfit =  np.max(fit), avelifepoint = np.mean(lifepoint))
        

        # saves stats of first gen
        file_aux  = open(experiment_name+'/enemy'+ str(enemies[2])+'results'+ str(i) +'.txt','a')
        file_aux.write('\n\ngen mean std max')
        print( '\n GENERATION '+str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
        file_aux.write('\n'+ str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6))   )
        file_aux.close()

        # gets best of first gen
        index = fit.index(np.max(fit))
        ind = Pop[index]
        best = tlbx.clone(ind)

        # saves file with the best solution
        np.savetxt(experiment_name+'/enemy'+ str(enemies[2])+'best'+ str(i) +'.txt', best)


        for n_gen in range(max_gens):
            n_gen += 1
            print('---------------------Generation {}-------------------------'.format(n_gen))
            
            # Parent selection
            offspring = tlbx.select(Pop)
            offspring = list(map(tlbx.clone, offspring))
            
            # goes of offspring list
            for parent1, parent2 in zip(offspring[::2], offspring[1::2]):
                
                # mutates first parent
                if random.random() < mut_prob:  
                    tlbx.mutate(parent1, sigma)
                    del parent1.fitness.values

                # mutates second parent
                if random.random() < mut_prob:  
                    tlbx.mutate(parent2, sigma)
                    del parent2.fitness.values

                # mates parents
                if random.random() < OffProb:
                    tlbx.mate(parent1, parent2, random.random())
                    del parent1.fitness.values
                    del parent2.fitness.values

            # evaluates new individuals in the pop
            new_ind = [ind for ind in offspring if not ind.fitness.valid]    
            fitns = list(map(tlbx.evaluate, new_ind))

            for ind, fit in zip(new_ind, fitns):
                ind.fitness.values = fit

            # replaces pop and retrives all fitnesses
            Pop[:] = tlbx.survival(offspring, len(Pop))
            fits = [ind.fitness.values[0] for ind in Pop]
        
            lifepoint = [ind.lifepoint for ind in Pop]

            # logs results
            log.record(gen = n_gen, meanfit = np.mean(fits), varfit = np.var(fits), stdfit = np.std(fits), maxfit =  np.max(fits), avelifepoint = np.mean(lifepoint))
            
            # save result
            file_aux  = open(experiment_name+'/enemy'+ str(enemies[2])+'results'+ str(i) +'.txt','a')
            print( '\n GENERATION '+str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6)))
            file_aux.write('\n'+ str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6))   )
            file_aux.close()
            

            # checks if best of gen is best ever
            if best.fitness.values < log[n_gen].get('maxfit'):
                
                # finds best
                index = fit.index(np.max(fit))
                ind = Pop[index]
                best = tlbx.clone(ind)
                
                # saves file with the best solution
                np.savetxt(experiment_name+'/enemy'+ str(enemies[2])+'best'+ str(i) +'.txt', best)

            # checks if meanfit keeps improving
            if log[n_gen].get('meanfit') <= log[n_gen - 1].get('meanfit') : 
                noimprovement += 1
            else :
                noimprovement = 0
            
            # if to long not improved execute doomsday
            if noimprovement > (max_gens//10):

                # deletes log of this gen and makes new pop and fitness
                del log[n_gen]
                tlbx.Doomsday(Pop, fits, sigma)
                fits = [ind.fitness.values[0] for ind in Pop]
                lifepoint = [ind.lifepoint for ind in Pop]
                noimprovement = 0

                # logs results
                log.record(gen = n_gen, meanfit = np.mean(fits), varfit = np.var(fits), stdfit = np.std(fits), maxfit =  np.max(fits), avelifepoint = np.mean(lifepoint))
            
                # save result
                file_aux  = open(experiment_name+'/enemy'+ str(enemies[2])+'results'+ str(i) +'.txt','a')
                print( '\n ~~~~~~~~~~DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOM~~~~~~~~~~')
                file_aux.write('\n'+ str(n_gen)+' '+str(round(log[n_gen].get('meanfit'),6))+' '+str(round(log[n_gen].get('stdfit'),6))+' '+str(round(log[n_gen].get('maxfit'),6))   )
                file_aux.close()
        
            Logs[i] = log
            
        


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
    

fig, pl = plt.subplots(3)

pl[0].plot(Logs[0].select('gen'), mean_average)
pl[0].set_ylabel('Average fit')
pl[1].plot(Logs[0].select('gen'), std_average)
pl[1].set_ylabel('Average std')
pl[2].plot(Logs[0].select('gen'), lifepoint_average)
pl[2].set_ylabel('Average life')
pl[2].set_xlabel('Generations')
fig.savefig('enemy3.png')


plt.show()
# average life points

print(log.select('meanfit'))
print(best.fitness.values)
