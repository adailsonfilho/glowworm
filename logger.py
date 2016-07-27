import numpy as np
import json

class Logger:

	def __init__(self):
		self.data = []
		self.fitness = []
		self.current_epoch = 0

		#epochs control
		self.data_acc = []
		self.fitness_acc = []
	
	def append_agent(self, features, fitness, epoch):

		if epoch > self.current_epoch:

			#salva dados da epoca passada
			self.data.append(self.data_acc)
			self.fitness.append(self.fitness_acc)

			#prepara para nova epoca
			self.data_acc = []
			self.fitness_acc = []

			self.current_epoch = epoch

		#salva dados de entrada de novo individuo
		self.data_acc.append(list(features))
		self.fitness_acc.append(fitness)

	def save_log(self, filename,data_only=False):

		log = {}
		if not data_only:
			header = {
				'dimension': len(self.data[0][0]),
				'epochs': len(self.data[0])
			}
			log['header'] = header

		log['data'] = self.data
		log['fitness'] = self.fitness

		if not filename.endswith('.swarm'):
			filename = filename +'.swarm'

		with open(filename,'w') as log_file:
			json_log = json.dumps(log)
			log_file.write(json_log)

		print('Log saved successfully as:',filename)


