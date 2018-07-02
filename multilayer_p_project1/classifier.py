import numpy as np

from mlp import *
from util import *
import data_handler


class MLPRegressor(MLP):

	def __init__(self, dim_in, dim_hid, dim_out, validation_data, validation_label):
		super().__init__(dim_in, dim_hid, dim_out, validation_data, validation_label)


	## functions

	def cost(self, targets, outputs): # new
		return np.sum((targets - outputs)**2, axis=0)

	def f_hid(self, x): # override
		return(1/(1 + np.exp(-x))) # sigmoid

	def df_hid(self, x): # override
		return self.f_hid(x)*(1 - self.f_hid(x)) # derivation of sigmoid

	def f_out(self, x): # override
		#return(1/(1 + np.exp(-x))) # sigmoid
		"""Compute the softmax of vector x in a numerically stable way."""
		shiftx = x - np.max(x)
		exps = np.exp(shiftx)
		return exps / np.sum(exps)		#softmax

	def df_out(self, x): # override
		#f_h = self.f_hid(x)*(1 - self.f_hid(x))
		#print(" fh sh ", f_h.shape)
		#return jac
		return self.f_out(x)*(1 - self.f_out(x)) # derivation of sigmoid


	def stablesoftmax(self, x):
		"""Compute the softmax of vector x in a numerically stable way."""
		shiftx = x - np.max(x)
		exps = np.exp(shiftx)
		return exps / np.sum(exps)
	
	def der_softmax(self, x):
		x = np.atleast_2d(x)
		J = - x[..., None] * x[:, None, :] # off-diagonal Jacobian
		iy, ix = np.diag_indices_from(J[0])
		J[:, iy, ix] = x * (1. - x) # diagonal
		return J.sum(axis=1) # sum across-rows for each sample

	## prediction pass

	def predict(self, inputs):
		outputs, *_ = self.forward(inputs)  # if self.forward() can take a whole batch
		# outputs = np.stack([self.forward(x)[0] for x in inputs.T]) # otherwise
		return outputs


	def early_stopping(self, train_errors, val_errors):
		quotient = 0
		#print('val errors ', val_errors[-1], min(val_errors))
		#print('pomer val errors ', (val_errors[-1] / min(val_errors)))
		gener_loss = 100 * ((val_errors[-1] / min(val_errors)) - 1)
		#print ('gener loss ', gener_loss)
		#print('train train_progress ', sum(train_errors) / (len(train_errors) * (min(train_errors))))
		#print('train errors ', train_errors)
		train_progress = 1000 * (sum(train_errors) / (len(train_errors) * (min(train_errors))) - 1)
		try:
			quotient = gener_loss / train_progress
		except:
			#print('was 0 ', gener_loss, '#', train_progress)
			quotient = 0
		#print('train_progress ', train_progress)
		#print('quotient ', quotient)
		return quotient
	
	## training
	def train(self, inputs, targets, alpha=0.1, eps=100, early_stop_slice_len = 10, quotient_lvl = 1, early_stopping = True):
		(_, count) = inputs.shape
		errors = []
		CEs = []
		REs = []
		valE = []
		temp_REs = []
		all_weights = []
		#print('inputs sh', targets.shape)
		for ep in range(eps):
			print('Ep {:3d}/{}: '.format(ep+1, eps), end='')
			E = 0
			RE = 0
			
			for i in np.random.permutation(count):
				x = inputs[:, i] # FIXME
				#print (x)
				#print ("x sh ", x.shape)
				d = targets[:, i] # FIXME
				#print(d)

				y, dW_hid, dW_mid, dW_out = self.backward(x, d)
				#print('dw hid shape ', dW_hid.shape)
				#print(d)
				#print(self.decode_argmax(y))
				#print(RE)
				#print(self.cost(d,y))
				#print('@@@@')
				E += self.cost(d,y)
				RE += self.decode_argmax(d) != self.decode_argmax(y)
				#print (RE)
				#print('####')
				self.W_hid += alpha * dW_hid # FIXME
				
				self.W_firstmid += alpha * dW_mid
				
				self.W_out += alpha * dW_out # FIXME

			all_weights.append((self.W_hid, self.W_firstmid, self.W_out))
			E /= count	#get percentage
			RE /= count
			temp_REs.append(RE)		# list of errors for last 'early_stop_slice_len' epochs
			if ep > 0 and (ep % early_stop_slice_len == 0) and early_stopping:	# we have done 'early_stop_slice_len' epochs, now we want check if we are overfitting
				#print('in early early_stopping')
				outputs = self.predict(self.validation_data)
				outputs = data_handler.Handler().decode_labels(outputs).T		#3,1600
				validation_error = self.evaluate_model(self.validation_label, outputs)
				#print('vali error ', validation_error)
				valE.append(validation_error)
				quotient = self.early_stopping(temp_REs, valE)
				
				if quotient > quotient_lvl:
					print('OVERFITTING , pls stop trainin at epoch ', ep, ' with train error ', temp_REs)
					#print('last validation err ', validation_error)
					#self.W_hid -= alpha * dW_hid # FIXME
					#self.W_out -= alpha * dW_out # FIXME
					#print('go "slice" episodes back, ep to goto: ', ep - early_stop_slice_len)
					#print('previous slice val err ', valE)
					self.W_hid = all_weights[ep - early_stop_slice_len][0]
					self.W_firstmid = all_weights[ep - early_stop_slice_len][1]
					self.W_out = all_weights[ep - early_stop_slice_len][2]
					return (errors, REs)
				temp_REs = []
			errors.append(E)		#append only after we sure early stop did not interrupt
			REs.append(RE)

			#print('E = {:.3f}'.format(E))
			#print(E)
			#print(RE)
			#print('RE = {:.5f}'.format(RE))
			print('CE = {:6.2%}, RE = {:.5f}'.format(E, RE))
	
		return (errors, REs)


	def decode_argmax(self, to_decode):
		max_index = to_decode.argmax(axis = 0)
		if max_index == 0:
			return 'A'
		if max_index == 1:
			return 'B'
		if max_index == 2:
			return 'C'
		
		
		
	def evaluate_model(self, targets, outputs):
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
		return (count_errors / data_count)
		