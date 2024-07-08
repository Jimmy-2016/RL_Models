
import numpy as np
import matplotlib.pyplot as plt


plt.plot(np.cumsum(np.random.normal(0.05, 1, 100)), lw=4)
plt.show()