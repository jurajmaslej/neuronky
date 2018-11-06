import numpy as np
import random

from hopfield import Hopfield
from util import *
import copy
## 1. load data

dataset = 'projekt_data'
#dataset = 'small'
# dataset = 'medium'

patterns = []

with open(dataset+'.txt') as f:
    count, width, height = [int(x) for x in f.readline().split()] # header
    dim = width*height
    
    for _ in range(count):
        f.readline() # skip empty line
        x = np.empty((height, width))
        #print('x shape ', x.shape)
        for r in range(height):
            x[r,:] = np.array(list(f.readline().strip())) == '#'

        patterns.append(2*x.flatten()-1) # flatten to 1D vector, rescale {0,1} -> {-1,+1}

util_setup(width, height)


model = Hopfield(dim)
model.train(patterns)

def flip_positions(input, positions):
	flipped = 0
	new_noise = input.copy()
	for pos in positions:
		new_noise[pos] = - new_noise[pos]
		flipped += 1
	return new_noise
	
def discrete_noise_correction(input):
	all_inputs = []
	for noise_amount in [0,7,14,21]:
		positions = []
		for i in range(0, noise_amount):
			positions.append(np.random.randint(0,35))
		noised_in = flip_positions(input, positions)
		all_inputs.append(noised_in)
	return all_inputs

def is_true_atractor(last_state):
	for pat in patterns:
		if np.array_equal(pat, last_state):
			return True
	return False

def atractor_equality(a1, all_others):
	
	a2_int = []
	a2_int_invert = []
	a1_int = np.frombuffer(copy.copy(a1))
	#print('a1 int ', a1_int)
	for a in all_others:
		a2_int.append(np.frombuffer(a))
		a2_int_invert.append(-1 * a2_int[-1])
	for a_int, a_int_invert in zip(a2_int, a2_int_invert): 
		if np.array_equal(a1_int, a_int):
			return a_int.tobytes()
		if np.array_equal(a1_int, a_int_invert):
			return a_int.tobytes()
	return None

def atractor_stats(last_states, end_types):
	true_a = 0
	false_a = 0
	cycle_a = 0
	frequency = dict()
	for state, end_type in zip(last_states,end_types):
		state_as_bytes = state.tobytes()
		
		# already in, no differences
		if state_as_bytes in frequency.keys():
			frequency[state_as_bytes] += 1
			
		# search for allowed differences, then +1
		minor_diff = atractor_equality(state_as_bytes, frequency.keys())
		if minor_diff is not None and state_as_bytes not in frequency.keys():
			frequency[minor_diff] += 1
			
		# was never seen
		else:
			if state_as_bytes not in frequency.keys():
				frequency[state_as_bytes] = 1
				plus1 = True
			
		if is_true_atractor(state):
			true_a += 1
		if end_type == 'cycled':
			cycle_a += 1
		if not is_true_atractor(state) and end_type != 'cycled':
			false_a += 1
			
	#most_frequent = dict()
	#most_frequent = {key: value for key, value in frequency.items() if value in sorted(set(frequency.values()), reverse = True)}
	most_frequent = sorted(frequency, key=frequency.get, reverse=True)
	
	
	most_frequent_states = [np.frombuffer(key, dtype = int) for key in most_frequent][:20]
	print('max frequent attractor had frequency ', frequency[most_frequent[0]])
	
	plot_states(most_frequent_states, title = 'mf.png')
	return ([true_a, false_a, cycle_a], most_frequent_states)

def all_patterns_discrete_nc():
	all_all_overlaps = []
	all_all_energy = []
	all_inputs = []
	all_all_states = []
	
	for pattern, letter in zip(patterns, ['A','B','X','0']):
		all_overlaps = []
		all_energy = []
		all_states = []
		all_noisy_input = discrete_noise_correction(pattern)
		for i, noisy_input in enumerate (all_noisy_input):
			state, energy, _ = run_sync_det(noisy_input)
			overlaps = compute_overlap(state, patterns)
			#plot_states(state, energy, patterns[i], title= letter + str(i*7) + 'states.png')		#fragile, if 5 noise level, need new index for patterns
			#plot_stats(overlaps, state, energy, letter=letter, noise=i*7)
			all_states.append(state)
			all_inputs.append(noisy_input)
			all_overlaps.append(overlaps)
			all_energy.append(energy)
			
		all_all_overlaps.append(all_overlaps)
		all_all_energy.append(all_energy)
		all_all_states.append(all_states)
	return (all_all_overlaps, all_energy)

def random_input():
	input = np.zeros(dim) # TODO random +1/-1 bits
	input = np.random.randint(2, size=dim)
	input = np.array([-1 if i == 0 else 1 for i in input])
	return input

def all_random_inputs(num_of_inputs, do_plotting= False):
	rand_inputs = []
	last_states = []
	end_types = []
	for i in range (0, num_of_inputs):
		rand_inputs.append(random_input())
		state, energy, end_type  = run_sync_det(rand_inputs[-1])
		overlaps = compute_overlap(state, patterns)
		if do_plotting:
			plot_states(state, energy, title= 'states_random_' + str(i) + '.png')
			plot_stats(overlaps, state, energy, title='random' + str(i) + '.png')
		last_states.append(state[-1])
		end_types.append(end_type)
	a_stats, most_frequent = atractor_stats(last_states, end_types)
	plot_atractor_stats(a_stats)
	print('distribution of attractors : True, Spurious, Cycle ', a_stats)
	
def compute_overlap(states, patterns):
	overlaps = []
	for pattern in patterns:
		overlp = []
		for st in states:
			overl = np.sum(pattern == st) / dim
			overlp.append(overl)
		overlaps.append(overlp)
	return overlaps
	
	
## 5. run the model
def run_sync_det(noisy_input):
	#print('run sync det')
	#plot_states([input], 'Random/corrupted input')

	# a) synchronous deterministic

	S, E, end_type = model.run_sync(noisy_input, eps=15,  beta = None)
	#plot_states(S, 'Synchronous run')
	return S, E, end_type #last state

#for all paterns, for all noises => check all
print('generating input from pattern with noise 0,7,14,21')
all_overlaps, all_energy = all_patterns_discrete_nc()
print('ran ok')
#5 random inputs
print('5 random binary inputs')
#all_random_inputs(5, True)
print('ran ok')
print('5000 random binary inputs')
all_random_inputs(5000, False)
print('ran ok')
