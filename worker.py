import neurokit2 as nk
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
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
import rabbitmq
import time

import os
from BioSeq import BioSequential
plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
plt.rcParams['font.size'] = 13
neutralDir = "./sourceData/neutral/"
happyDir = "./sourceData/happy/"
fearDir = "./sourceData/fear/"
#Users data
neutral = BioSequential(neutralDir,show=False)

#System emotions
fear = BioSequential(fearDir,show=False)
happy = BioSequential(happyDir,show=False)

fear.resampleTo(1000)
neutral.resampleTo(1000)
happy.resampleTo(1000)

fear.load("0")
#happy.load("0")

#3 segments 
while True:
    out = fear.predict(neutral.getTrain()[:10])
    output = dict()
    output["data"] = out.tolist()
    output["id"] = "0"
    output["anotherAttr"] = "none"
    jsonOutput = json.dumps(output)
    rabbitmq.send("localhost","0",jsonOutput)
    time.sleep(5)
# print(out)
# x = range(0, len(out[0][0]))

# for cycle in out:
#     plt.plot(x, cycle[0])
# plt.show()

# for cycle in out:
#     plt.plot(x, cycle[1])
# plt.show()

# for cycle in out:
#     plt.plot(x, cycle[2])
# plt.show()

# for cycle in out:
#     plt.plot(x, cycle[3])
# plt.show()