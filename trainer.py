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

#neutral = BioSequential(neutralDir,show=False)
fear = BioSequential(fearDir,show=False)
fear.resampleTo(1000)
#neutral.resampleTo(1000)
fear.initAutoEncoder()
fear.fit()
fear.save("0")
#out = fear.predict(neutral.getTrain()[:3])

