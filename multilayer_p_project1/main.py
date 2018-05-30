import data_handler
import numpy as np
from mlp import *
from classifier import *
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import itertools

from json import dumps
class Controller:
	
	def __init__(self):
		print("Hi, how are you? :) ")
		data = data_handler.Handler()
		
		# transpose matrices for conventions
		self.estimation_data = data.estimation_data.T
		self.validation_data = data.validation_data.T
		self.test_data = data.test_data.T
		
		self.estimation_label = data.estimation_label.T
		self.validation_label = data.validation_label.T
		self.test_label = data.test_label.T
		self.test_label_raw = data.test_label_raw.T		#so that i wont need to dcode once again
		self.validation_label_raw = data.validation_label_raw
		#print("estim data sh", self.estimation_data.shape)
		#print("estim lab sh", self.estimation_label.shape)
		#print(self.estimation_data.T)
		#print(self.estimation_data)
		
		self.estimation_errors = dict()
		self.validation_errors = dict()
		
		self.normalize_data()
		
		#model, best_key = self.test_model_params()
		model, outputs_create_m = self.create_model()
		#error, outputs = self.test_on_test_data(model = None, best_key = best_key)
		#outputs = np.array([[0,0,1],[1,0,0],[0,1,0],[1,0,0]])
		#print(outputs)
		#self.confusion_matrix(outputs, best_key, 'test')
		
		#self.save_val_and_est_errors()
		
	def save_val_and_est_errors(self):
		#est_errors_f = open('estimation_errors.txt', 'w')
		with open('estimation_errors.txt', 'w') as file:
			file.write(dumps(self.estimation_errors, sort_keys=True,indent=4, separators=(',', ': ')))
			
		with open('validation_errors.txt', 'w') as file:
			file.write(dumps(self.validation_errors, sort_keys=True,indent=4, separators=(',', ': ')))
		
	def test_on_test_data(self, model= None, best_key= None):
		print('best key used ', best_key)
		if model is None and best_key is None:
			return 'user must specify model or hyperparams'
		if model is None:
			epochs, alpha, neurons = best_key.split(':')
			epochs, alpha, neurons = [int(epochs), float(alpha), int(neurons)]
			#print('est sh ', self.estimation_label.shape)
			#print('val d ', self.validation_label.shape)
			train_fortest_data = np.hstack((self.estimation_data, self.validation_data))
			train_fortest_label =  np.hstack((self.estimation_label, self.validation_label))
			print (train_fortest_label.shape)
			
			model =  MLPRegressor(train_fortest_data.shape[0], neurons, train_fortest_label.shape[0], self.validation_data, self.validation_label)		#problem with validation data already seen
			
			trainREs = model.train(self.test_data, self.test_label, alpha= alpha, eps= epochs, early_stopping = False)
		
		print('on test data')
		outputs = model.predict(self.test_data)
		print('test outputs shape ', outputs.shape)
		outputs = data_handler.Handler().decode_labels(outputs).T		#3,1600
		error = self.evaluate_model_tot_errors(self.test_label, outputs)
		print('test error ', error)
		print('test outputs shape ', outputs.shape)
		plot_reg_density('Density', self.test_data, self.test_label, outputs, block=False)
		plot_errors('Model loss', trainREs[0], block=False)
		
		return (error, outputs)
	
	def confusion_matrix(self, outputs, params, target_set):
		decoded_outputs = []
		for i in outputs.T:
			#print('# ',self.decode_argmax(i))
			decoded_outputs.append(self.decode_argmax(i))
		#print('test lab shape ', self.test_label_raw.shape)
		#print('dec outputs len ', len(decoded_outputs))
		
		if target_set == 'validation':
			conf_matrix = confusion_matrix(self.validation_label_raw, decoded_outputs)
		else:
			conf_matrix = confusion_matrix(self.test_label_raw, decoded_outputs)
			params += '_final_testset_'
		cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
        
		cmap=plt.cm.Blues
		class_names= ['A', 'B', 'C']
        
		
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title('confusion_matrix ' + params)
		#plt.colorbar()
		tick_marks = np.arange(len(class_names))
		plt.xticks(tick_marks, class_names, rotation=45)
		plt.yticks(tick_marks, class_names)

		fmt = '.2f'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.savefig(params + 'conf_matrix.png')
		plt.close()
		#print('confusion_matrix ')
		#print(conf_matrix)
		#for i in self.test_label_raw:
		#	print(i)
		#should_a_was_b = 
		#for in in range(0,len(self.test_label_raw):
			
		
	def test_model_params(self):
		eps = np.arange(10,230,100)
		learn_rates = [0.12, 0.16]
		num_neurons = [10, 20, 25]
		created_models = dict()
		min_error = 10000
		best_key = ''
		prev_key = ''
		best_model = None
		for epoch in eps:
			for alpha_num in learn_rates:
				for neurons in num_neurons:
					#create_model
					model = MLPRegressor(self.estimation_data.shape[0], neurons, self.estimation_label.shape[0], self.validation_data, self.validation_label)
					trainREs = model.train(self.estimation_data, self.estimation_label, alpha= alpha_num, eps= epoch, early_stop_slice_len = 5)
					key = str(epoch) + ':' + str(alpha_num) + ':' + str(neurons)
					created_models[key] = model
					#compute validation error
					outputs = model.predict(self.validation_data)
					outputs = data_handler.Handler().decode_labels(outputs).T		#3,1600
					error = self.evaluate_model_tot_errors(self.validation_label, outputs)
					
					self.validation_errors[key] =  '{:.4f}'.format(error * 100) + '%'
					self.estimation_errors[key] = '{:.4f}'.format(trainREs[0][-1] * 100) + '%' 
					
					print('validation err for key ', key, ' err: ', error)
					if error < min_error:		#not <= but < to favour simpler models
						min_error = error
						best_key =  prev_key
						best_model = model
						print('new best found ', best_key ,', with error ', error)
					self.confusion_matrix(outputs, best_key, 'validation')
					prev_key = key
					print('key tried ', key)
		print('best key overall was ', best_key, '. With error: ', min_error)
		return (model, best_key)
						
		
	def normalize_data(self):
		self.estimation_data -=  np.mean(self.estimation_data, axis = 1, keepdims = True)
		self.estimation_data /= np.std(self.estimation_data, axis = 1, keepdims = True)
		
		self.validation_data -=  np.mean(self.validation_data, axis = 1, keepdims = True)
		self.validation_data /= np.std(self.validation_data, axis = 1, keepdims = True)
		
		self.test_data -=  np.mean(self.test_data, axis = 1, keepdims = True)
		self.test_data /= np.std(self.test_data, axis = 1, keepdims = True)
		
	def create_model(self, hidden_layers = 1, neurons = 20, alpha = 0.1, eps = 100 ):
		model = MLPRegressor(self.estimation_data.shape[0], 20, self.estimation_label.shape[0], self.validation_data, self.validation_label)
		trainREs = model.train(self.estimation_data, self.estimation_label, alpha=0.07, eps=100, early_stop_slice_len = 8)
		outputs = model.predict(self.validation_data)
		outputs = data_handler.Handler().decode_labels(outputs).T		#3,1600
		self.evaluate_model_tot_errors(self.validation_label, outputs)
		
		#return model
	
		#plot_reg_density('Density', self.validation_data, self.validation_label, outputs, block=False)
		#plot_errors('Model loss', trainREs[0], block=False)
		return (model, outputs)
	
	
	def evaluate_model_tot_errors(self, targets, outputs):
		targets = targets.T
		outputs = outputs.T
		#print(targets.shape)
		#print(outputs.shape)
		data_count,_ = outputs.shape
		count_errors = 0
		for i in range(0, data_count):
			if (outputs[i] == targets[i]).all() == False:
				count_errors += 1
				#print(self.validation_label.T[i])
				#print('#')
				#print(outputs.T[i])
				#print('@@@@')
		#print('num of count_errors in prediction ', count_errors)
		#print('count_errors / num of samples ', count_errors/data_count)
		return count_errors / data_count
	
	def decode_argmax(self, to_decode):
		max_index = to_decode.argmax(axis = 0)
		if max_index == 0:
			return 'A'
		if max_index == 1:
			return 'B'
		if max_index == 2:
			return 'C'
	
c = Controller()