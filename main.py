import neurokit2 as nk
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import arff
import pandas2arff as pa
# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt
import keras
from keras import layers
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras

import os
from BioSeq import BioSequential
plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
plt.rcParams['font.size'] = 13
neutralDir = "./sourceData/neutral/"
happyDir = "./sourceData/happy/"
fearDir = "./sourceData/fear/"

neutral = BioSequential(neutralDir,show=False)
fear = BioSequential(fearDir,show=False)
fear.resampleTo(1000)
neutral.resampleTo(1000)
fear.initAutoEncoder()
fear.fit()
out = fear.predict(neutral.getTrain()[:3])
print(out)
x = range(0, len(out[0][0]))

for cycle in out:
    plt.plot(x, cycle[0])
plt.show()

for cycle in out:
    plt.plot(x, cycle[1])
plt.show()

for cycle in out:
    plt.plot(x, cycle[2])
plt.show()

for cycle in out:
    plt.plot(x, cycle[3])
plt.show()