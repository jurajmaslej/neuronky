import itertools
import numpy as np

from util import *


class Hopfield():

	def __init__(self, dim):
		self.dim  = dim


	def train(self, patterns):
		#print('pattern ', len(patterns[0]))
		self.W = np.zeros((self.dim, self.dim))
		for i in range(len(patterns)):
			self.W += np.outer(patterns[i], patterns[i])
		#print('weights sh ', self.W.shape)
		# TODO compute weight matrix analytically
		np.fill_diagonal(self.W, 0)

	def energy(self, s):
		temp = 0
		for j in range(0, self.dim):
			for i in range(0, self.dim):
				if i != j:
					temp += self.W[j, i] * s[i] * s[j]
		return (-1/2) * temp # TODO compute the energy of a state


	# compute next state
	# - if neuron=None, synchronous dynamic: return a new state for all neurons
	# - otherwise, asynchronous dynamic: return a new state for the `neuron`-th neuron
	def forward(self, s, neuron=None):
		#print('s shape ', s.shape)
		net = np.dot(self.W, s) # FIXME
		
		if neuron is not None:
			if self.beta is not None: # stochastic
				probs = 1 / (1 + np.exp( - net / self.beta))
				for i in range(0, len(probs)):
					dice = np.random.uniform(0,1)
					if dice < probs[i]:
						net[i] = 1
					else:
						net[i] = -1
				return net[neuron]
			else: # deterministic
				return np.sign(net+0.0001)[neuron] # FIXME
			
		#print('net ', net.shape)
		if self.beta is not None: # stochastic
			probs = 1 / (1 + np.exp( - net / self.beta))
			for i in range(0, len(probs)):
				dice = np.random.uniform(0,1)
				if dice < probs[i]:
					net[i] = 1
				else:
					net[i] = -1
			return net
				
		else: # deterministic
			return np.sign(net +0.0001) # FIXME


	def run_sync(self, x, eps=None, beta=None):
		s = x.copy()
		e = self.energy(s)
		S = [s]
		E = [e]
		self.beta = beta # set temperature for stochastic / None for deterministic

		for _ in intertools.count() if eps is None else range(eps): # enless loop (eps=None) / up to eps epochs
			s = self.forward(s, neuron=None) # update ALL neurons
			e = self.energy(s)
			S.append(s)
			E.append(e)
			if np.array_equal(S[-1], S[-2]) and len(S) > 2: # if last two are equal, 
				return S, E, 'repetitive'
			#print('state len ', len(S))
			#print('chceck to ', len(S[:-2]))
			for state in S[:-2]:
				if np.array_equal(state, S[-1]):
					#print('cycled')
					return S, E, 'cycled'
		return S, E, 'eps_runout' # if eps run out

	def run_async(self, x, eps=None, beta_s=None, beta_f=None, row=1, rows=1, trace=False):
		s = x.copy()
		e = self.energy(s)
		E = [e]

		title = 'Running: asynchronous {}'.format('stochastic' if beta_s is not None else 'deterministic')

		for ep in range(eps):
			if beta_s: # stochastic => temperature schedule
				self.beta = beta_s * ( (beta_f/beta_s) ** (ep/(eps-1)) )
				print('Ep {:2d}/{:2d}:  stochastic, beta = 1/T = {:7.4f}'.format(ep+1, eps, self.beta))
			else: # deterministic
				self.beta = None
				print('Ep {:2d}/{:2d}:  deterministic'.format(ep+1, eps))

			for i in np.random.permutation(self.dim):
				s[i] = self.forward(s, neuron=i) # update ONE neuron
				e = self.energy(s)
				E.append(e)

				if trace:
					plot_state(s, errors=E, index=i, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=False)
					redraw()

			if not trace:
				plot_state(s, errors=E, index=None, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=False)
				redraw()

			# terminate deterministic when stuck in a local/global minimum (loops generally don't occur)

			if self.beta is None:
				if np.all(self.forward(s) == s):
					break
