from glowworm import gso
from deap import benchmarks

if __name__ == '__main__':

	def fitness(candidate):
		return 1/(benchmarks.ackley(candidate)[0]+1)

	gso(individuals_number=100, dim=10, func_obj=fitness,epochs=5000,step_size=0.1, dims_lim = [-15,15])