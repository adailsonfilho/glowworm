from glowworm import gso
from deap import benchmarks

if __name__ == '__main__':

	def fitness(candidate):
		# return 1/(benchmarks.schwefel(candidate)[0]+1)
		return 1/(benchmarks.ackley(candidate)[0]+1)

	gso(agents_number=80, dim=10, func_obj=fitness, epochs=300, step_size=0.5, random_step =True, virtual_individual = True, dims_lim = [-15,15])