# imports framework
import sys
sys.path.insert(0, 'evoman')

from environment import Environment
from demo_controller import player_controller

import glob, os
from deap import base, creator, tools
import random
import numpy as np

experiment_name = 'spot_evolution_testing'
if not os.path.exists(experiment_name):
	os.makedirs(experiment_name)

# import arguments given by the user
arguments = sys.argv[1]
spot_parameters = eval(arguments.split()[0])

# enemy(ies) to play against
enemy = 2

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
pop_size = 10
n_gens = 3
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
upper_w = 1
lower_w = -1

# sigma for normal dist, tau constant,  mut_prob prop mutation for individu
sigma = 1
tau = 1/np.sqrt(n_weights)
mate_prob = spot_parameters[0]
mut_prob = spot_parameters[1]
average_pops = []
std_pops = []
best_per_gen = []
best_overall = 0

log = tools.Logbook()
tlbx = base.Toolbox()
env.state_to_log() # checks environment state


# register and create deap functions and classes
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
	evaluate.counter+=1
	file = open('number_of_evaluations.txt', 'w')
	file.write(str(evaluate.counter))
	file.close()
	f, _, _, _ = env.play(pcont=x)
	return f,

evaluate.counter = 0

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
	len_matingpop = 3 * len(pop)

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

# register deap functions
register_deap_functions()

# initializes population at random
pop = tlbx.population()
pop_fit = [tlbx.evaluate(individual) for individual in pop]

best = np.argmax(pop_fit)
std = np.std(pop_fit)
mean = np.mean(pop_fit)

average_pops.append(mean)
std_pops.append(std)
best_per_gen.append(pop_fit[best])

np.savetxt('test.txt', [1])

for ind, fit in zip(pop, pop_fit):
	ind.fitness.values = fit

means_of_n_gens = np.zeros(n_gens)
for n_gen in range(n_gens):

	offspring = tlbx.select(pop, pop_size)
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

	new_pop = [check_bounds(ind, lower_w, upper_w) for ind in new_pop]
	new_pop_fitness = [tlbx.evaluate(ind) for ind in new_pop]

	for ind, fit in zip(new_pop, new_pop_fitness):
		ind.fitness.values = fit

	try:
		pop[:] = tlbx.survival(new_pop, pop_size)
		pop_fit = np.array([ind.fitness.values for ind in pop])


		best = np.argmax(pop_fit)
		std = np.std(pop_fit)
		mean = np.mean(pop_fit)
		means_of_n_gens[n_gen] = -mean

	except ValueError:
		f = open('error_log.txt', 'a')
		f.write('ValueError\n')
		f.close()
		means_of_n_gens[n_gen] = 100

	# average_pops.append(mean)
	# std_pops.append(std)
	# best_per_gen.append(pop_fit[best])

	# if pop_fit[best] > best_overall:
		# np.savetxt(experiment_name + '/best_solution_enemy{}.txt'.format(enemy), pop[best])
print(means_of_n_gens.mean())
# np.savetxt(experiment_name + "/average_gen_enemy{}.txt".format(enemy), average_pops)
# np.savetxt(experiment_name + "/std_gen_enemy{}.txt".format(enemy), std_pops)
# np.savetxt(experiment_name + "/best_of_gen_enemy{}.txt".format(enemy), best_per_gen)