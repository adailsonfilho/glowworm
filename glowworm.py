import numpy as np
import ipdb

"""
GLOWWORM SWARM OPTIMIZATION (GSO)

Implementation:
- Author: Adailson de Castro Queiroz Filho,
- At: Jul 2016

GSO Author:
- TITLE: Glowworm swarm optimization for simultaneous capture of multiple local optima of multimodal functions
- DOI: 10.1007/s11721-008-0021-5
- Author: K.N. Krishnanand · D. Ghose

GSO Basic documentation:
	agents_number:	(int) number of individuals
	dim:			(int) dimensions of the problem
	func_obj:		(function) function that returns a fitness for a given individual
	epochs:			(int) maximos number of epochs
	step_size:		(double) size of the step of each individual (fixed when 'random_step' == False, and set as the max value when 'random_step' == False)
	dims_lim:		(list of int, with 2 positions) a list of two values, determining boundaries of the dimensions (considering the same for all dimensions in this version, should be updated for each one in a near future)
	random_step:	True or False - Determine if the step is fixed or exists a random between a minium value and the step size

	triggers:
		- individual_updated	= runs for each time an individual is updated
		- program_ends 			= runs when the program ends

"""

def gso(agents_number, dim, func_obj, epochs, step_size, r0, rs, b, k_neigh, dims_lim = [-1,1], random_step=False, virtual_individual = False, individual_updated=None, program_ends = None):
	
	assert len(dims_lim) == 2
	assert type(agents_number) == int
	assert type(dim) == int

	"""
	PARAMETTERS
	"""
	#RANGE of sight
	range_init =r0
	# range_min = 0.1
	range_boundary = rs

	#LUCIFERIN
	luciferin_init = 5
	luciferin_decay = 0.4
	luciferin_enhancement = 0.6

	#POSITION
	# step_size = 0.1

	#Neighbors
	beta = b

	"""
	AGENTES INITIALIZATION
	"""

	#random initiation
	_min = dims_lim[0]
	_max = dims_lim[1]
	glowworms = np.random.uniform(_min,_max,[agents_number,dim])

	#distances
	distances = np.zeros([agents_number,agents_number])

	#luciferin
	luciferins = np.zeros(agents_number)
	luciferins += luciferin_init
	
	#range of sigth
	ranges = np.zeros(agents_number)
	ranges += range_init

	#dynamic step size parametters
	# variance_direction = 0
	# consecutive_fails = 0
	# min_evolution = 0.01

	"""
	LUCIFERIN UPDATE PHASE
	"""
	def luciferin_update(last_luciferin,fitness):
		l = ((1-luciferin_decay)*last_luciferin) + (luciferin_enhancement*fitness)
		return l

	"""
	POSITION UPDATE PHASE
	"""
	def find_neighbors(glowworm_index):

		i = glowworm_index
		neighbors_index = []

		#look for all the neighbors in a triangle distance matrix
		for k in range(agents_number):

			dist = get_distance(i,k)
			#if it is in it's range of sigth and it's brightness is higher
			if dist != 0 and dist <= ranges[i] and luciferins[i] < luciferins[k] :
				neighbors_index.append(k)

		return neighbors_index


	def follow(glowworm_index,neighbors_index):

		#current luciferin - li
		li = luciferins[glowworm_index]

		#luciferin of all neighbors (lj or lk)
		sum_lk = sum([luciferins[k] for k in neighbors_index])

		#calc probabilties for each neighbor been followed
		probs = np.array([luciferins[j]-li for j in neighbors_index])
		probs /= sum_lk - (len(probs)*li)

		#calc prob range
		acc = 0
		wheel = []
		for prob in probs:
			acc += prob
			wheel.append(acc)

		wheel[-1] = 1

		#randomly choice a value for wheel selection method
		rand_val = np.random.random()
		following = None
		for i, value in enumerate(wheel):
			if rand_val <= value:
				following = i

		return neighbors_index[following]

	def position_update(i, j):
		glowworm = glowworms[i]

		toward = None
		if type(j) == int:
			toward = glowworms[j]
		elif type(j) == type(np.array([])):
			toward = j

		norm = np.linalg.norm(toward-glowworm)

		if norm == 0 or np.isnan(norm):
			norm = step_size

		# if dynamic_step_size:
		# 	step_size = 

		if random_step:
			step = step_size*np.random.random()
			if step < 0.01:
				step = 0.01
		else:
			step = step_size

		new_position = glowworm + step_size*(toward-glowworm)/norm

		#update distmatrix for all associated cells (not all matrix)
		for k in range(agents_number):

			if max(k,i) == k:
				distances[i][k] = np.linalg.norm(new_position-glowworms[k])
			elif max(k,i) == i:
				distances[k][i] = np.linalg.norm(new_position-glowworms[k])

		return new_position

	def range_update(glowworm_index, neighbors):
		return min(range_boundary,max(0,ranges[glowworm_index] + (beta*(k_neigh-len(neighbors)))))

	def virtual_glowworm(glowworm_index):
		glowworm = glowworms[glowworm_index]
		virtual = np.random.uniform(_min,_max,[1,dim])
		virtual = glowworm + ranges[glowworm_index]*(virtual-glowworm)/np.linalg.norm(virtual-glowworm)
		return virtual
		

	def get_distance(i,j):
		if max(j,i) == j:
			return distances[i][j]
		elif max(j,i) == i:
			return distances[j][i]
		else:
			return 0.0
	

	"""
	EXECUTION
	"""
	fitness = np.zeros(agents_number)
	best_fitness_history = []

	#initialize distance matrix (over main diagonal, only)
	for i in range(agents_number):
		for j in range(agents_number):
			if max(i,j) == j:
				distances[i][j] = np.linalg.norm(glowworms[i]-glowworms[j])
			elif max(i,j) == i:
				distances[j][i] = np.linalg.norm(glowworms[i]-glowworms[j])

	for epoch in range(epochs):

		epoch_fitness_history = []

		#update all glowworms luciferin
		for i in range(agents_number):

			li = luciferins[i]
			if epoch == 0:				
				fitness[i] = func_obj(glowworms[i])

			luciferins[i] = luciferin_update(li,fitness[i])
			epoch_fitness_history.append(fitness[i])

		best_fitness_history.append(max(epoch_fitness_history))

		#movement phase
		for i in range(agents_number):

			#find best neighbors
			neighbors = find_neighbors(i)

			change = False
			if len(neighbors) > 0:

				toward_index = follow(i,neighbors)
				glowworms[i] = position_update(i,toward_index)
				change = True

			elif virtual_individual:

				virtual = virtual_glowworm(i)
				glowworms[i] = position_update(i,virtual)
				change = True
			else:
				print('fez nada')

			#atualiza fitness, mas nao luciferina
			if change:
				fitness[i] = func_obj(glowworms[i])

			if individual_updated is not None:
				individual_updated(glowworms[i], fitness[i], epoch)

			ranges[i] = range_update(i,neighbors)

		print(epoch,'> best:',best_fitness_history[-1])

	if program_ends is not None:
		program_ends()
