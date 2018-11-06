import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Handler:
	
	'''
	todo with data:
		0. load train and test
		1. shuffle both of them
		2. split test to estimation and validation datasets
		3. split estimation and validation to input_data and labels
		xx. Normalization/rescalling
		
	variables:
		train_f = np array, all from train_file, shuffled in __init__
		test_f = np array, all from test_file, shuffled in __init__
		estimation = np array, 80% of train_f 
		validation = np array, 20% of test_f
		estimation_data = data from estimation
		estimation_label = labels from estimation
		validation_data = data from validation
		validation_label = labels from validation
	'''
	
	def __init__(self, train_file = "train_data.txt", test_file = "test_data.txt"):
		train_f = np.genfromtxt(train_file, delimiter= ' ', dtype = str, skip_header = 1)
		test_f = np.genfromtxt(test_file, delimiter= ' ', dtype = str, skip_header = 1)
		
		#shuffle data
		np.random.shuffle(train_f)
		np.random.shuffle(test_f)
		#/shuffle data
		
		estimation, validation = self.eighty_twenty_split(train_f)
		self.estimation_data = estimation[:, :2].astype(float)
		self.estimation_label = estimation[:, 2]
		
		self.validation_data = validation[:, :2].astype(float)
		self.validation_label = validation[:, 2]
		
		'''
		c = 0
		c_t = 0
		for i in self.validation_label.T:
			if i == 'C':
				c += 1
		for i in self.estimation_label.T:
			if i == 'C':
				c_t += 1
		print('ct  ', c_t / (8000 - 1600) )
		print('c val ', c / 1600)
		'''
		
		self.test_f = test_f
		self.test_data = test_f[:, :2].astype(float)
		self.test_label = test_f[:, 2]
		self.test_label_raw = self.test_label
		self.validation_label_raw = self.validation_label
		
		self.estimation_label = self.encode_labels(self.estimation_label).astype(float)
		self.validation_label = self.encode_labels(self.validation_label).astype(float)
		self.test_label = self.encode_labels(self.test_label).astype(float)
		
	def split_label(self, data):
		input_data = data[:,:2]
		labels = np.atleast_2d(data[:,2]).T
		return (input_data, labels)
		
	def eighty_twenty_split(self, data):
		splitter = int(data.shape[0]*0.8)
		return data[:splitter],data[splitter:]
	
	def encode_labels(self, data):
		outputs = np.array([0,0,0])
		for i in data:
			if i[0] == 'A':
				#print (outputs.shape)
				outputs = np.vstack((outputs, np.array([1,0,0])))
			if i[0] == 'B':
				outputs = np.vstack((outputs, np.array([0,1,0])))
			if i[0] == 'C':
				outputs = np.vstack((outputs, np.array([0,0,1])))
        
		outputs = outputs[1:,:]
		return outputs
	
	def decode_labels(self, data):
		outputs = np.array([0,0,0])
		for i in data.T:
			#print (i)
			max_index = i.argmax()
			#print('max ind ', max_index)
			if max_index == 0:
				#print (outputs.shape)
				outputs = np.vstack((outputs, np.array([1,0,0])))
			if  max_index == 1:
				outputs = np.vstack((outputs, np.array([0,1,0])))
			if  max_index == 2:
				outputs = np.vstack((outputs, np.array([0,0,1])))
			#print(outputs)
		outputs = outputs[1:,:]
		#print('outputs ',outputs.shape)
		#for i in outputs:
		#	print (i)
		return outputs
	
	def all_data_to_floats(self):
		pass
	
	def train_x_data(self, file_obj):
		pass
	
	def train_y_data(self):
		pass
	

class Normalization:
	
	def __init__(self, data):
		self.data = data
#h = Handler("train_data.txt", "test_data.txt")
