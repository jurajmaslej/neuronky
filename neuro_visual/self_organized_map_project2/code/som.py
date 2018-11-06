import numpy as np

class SOM():

	def __init__(self, dim_in, n_rows, n_cols, inputs=None):
		self.dim_in = dim_in
		self.n_rows = n_rows
		self.n_cols = n_cols
		print('dim in ', dim_in)
		print('n rows ', n_rows)
		print('n cols ', n_cols)
		self.weights  = np.random.randn(n_rows, n_cols, dim_in)
		print('w shape ', self.weights.shape)
		print('inputs shape ', inputs.shape)
		
		# add class label for weights
		self.class_label  = np.zeros((n_rows, n_cols))
		self.count_activations = np.zeros((n_rows, n_cols))
		self.quant_err = []
		self.avg_quant_err = []
		self.adjustment = []
		self.avg_adjustment = []
		
		for dim in range(self.dim_in):		# scale weights
			self.weights[:,:, dim] += np.mean(inputs[dim,:])
        
	def L_2_euclid(self, X, Y):
		d = np.sqrt(((X[0] - Y[0])**2 + (X[1] + Y[1])**2)) 
		return d # TODO

	def winner(self, x):
		win_r, win_c, win_d = -1, -1, float('inf')
		local_min = 1000000
		min_dst = 100000
		for r in range(self.n_rows):
			for c in range(self.n_cols):
				local_min = np.linalg.norm(self.weights[r,c] - x)
				if win_d > local_min:
					min_dst = abs(self.weights[r,c] - x)
					win_d = local_min
					win_r = r
					win_c = c
		
		self.class_label[win_r, win_c] = x[-1]
		self.count_activations[win_r, win_c] += 1  
		return win_r, win_c


	def train(self, inputs, discrete=True, metric=lambda x,y:0, alpha_s=0.01, alpha_f=0.001, lambda_s=None,
			lambda_f=1, eps=100, in3d=True):
		(_, count) = inputs.shape

		for ep in range(eps):
			
			self.count_activations = np.zeros((self.n_rows, self.n_cols))	# only count for one epoch
			
			exponent = (ep - 1)/ (eps - 1)
			alpha_t = alpha_s * ((alpha_f/alpha_s) ** exponent)
			lambda_t = lambda_s * ((lambda_f/lambda_s) ** exponent)
			print()
			print('Ep {:3d}/{:3d}:'.format(ep+1,eps))
			print('  alpha_t = {:.3f}, lambda_t = {:.3f}'.format(alpha_t, lambda_t))

			for i in np.random.permutation(count):
				x = inputs[:,i]
				#print( 'x :',  x)
				win_r, win_c = self.winner(x)

				for r in range(self.n_rows):
					for c in range(self.n_cols):
						d = metric((r, c), (win_r, win_c)) # FIXME
						self.quant_err.append(d)
						if discrete:
							h = lambda_t > d
						else:
							h = np.exp(- ((d**2) / lambda_t**2)) #gausian neighbourhood # FIXME
						adjustment = alpha_t * (x - self.weights[r, c]) * h
						self.adjustment.append(abs(adjustment))
						self.weights[r,c] += adjustment # FIXME
			
			self.avg_quant_err.append(np.mean(self.quant_err))
			self.quant_err = []
			self.avg_adjustment.append(np.mean(self.adjustment))
			self.adjustment = []
