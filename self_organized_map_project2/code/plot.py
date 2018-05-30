import numpy as np 
import matplotlib.pyplot as plt

H = np.array([[3, 3, 3, 3],
 [3, 3, 3, 3],
 [2, 2, 2, 2],
 [2, 2, 2, 2],
 [2, 2, 2, 2],
 [2, 2, 2, 2],
 [2, 2, 2, 2],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]
)

Hwrong = np.array([[3, 3, 3, 3],
 [3, 3, 3, 3],
 [2, 2, 2, 2],
 [2, 2, 2, 2],
 [1, 1, 1, 1],
 [2, 2, 2, 2],
 [2, 2, 2, 2],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]
)
plt.imshow(Hwrong)
plt.show()
plt.imshow(H)
plt.show()