import numpy as np
import csv

class Data_loader():
	
	def __init__(self, filename):
		train_f = np.genfromtxt(filename, delimiter= '	', dtype = float)
		#print(train_f)
		#print (train_f.shape)
		self.dataset = train_f
		
	def normalize_data(self):
		self.labels = np.atleast_2d(self.dataset[:, 7].astype(float))
		self.dataset = self.dataset[:, :8].astype(float)		# gettin rid of class here
		self.dataset -=  np.mean(self.dataset, axis = 1, keepdims = True)
		self.dataset /= np.std(self.dataset, axis = 1, keepdims = True)
		
		#self.dataset = np.concatenate((self.dataset, self.labels.T),axis = 1)
		#print ('dset ', self.dataset)
			
d = Data_loader('dataset.txt')
