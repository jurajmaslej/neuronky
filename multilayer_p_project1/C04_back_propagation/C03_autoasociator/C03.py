import numpy as np

from linear import *
from noise import *
from util import *


## 1. load data

data = np.loadtxt('faces.dat')
(dim, count) = data.shape

show_images(data, 'Originals')


## 2. corrupt

corrupted = cutout_rows(data, 20, 35)
# corrupted = noise_gaussian(data, 50)
#corrupted = noise_salt_and_pepper(data, 0.25)

show_images(corrupted, 'Corrupted')


## 3. train analytically and reconstruct

model = LinearAutoassociator(dim, count)

model.analytical(data[:,:6])

reconstructed = model.reconstruct(corrupted)

show_images(reconstructed, 'Reconstruction (analytical=1..6)')


## 4. train iteratively and reconstruct

model.iterative(data[:,6:])

reconstructed = model.reconstruct(corrupted)

show_images(reconstructed, 'Reconstruction (analytical=1..6, iterative=7..9)')
print("OK")

## 5. novelty detection

novelty = model.novelty(corrupted)

show_images(novelty, 'Novelty detection')
