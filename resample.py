import numpy as np
from scipy import signal
x = np.linspace(0, 10, 30, endpoint=False)
y = np.cos(-x**2/6.0)
f = signal.resample(y, 10)
xnew = np.linspace(0, 10, 10, endpoint=False)

import matplotlib.pyplot as plt
plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')
plt.legend(['data', 'resampled'], loc='best')
plt.show()