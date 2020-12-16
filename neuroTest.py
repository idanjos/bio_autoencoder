# Load NeuroKit and other useful packages
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import arff
import pandas2arff as pa
# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from keras.models                   import Sequential
from keras.layers                   import Dense 
import keras
import numpy as np 
import os
plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
plt.rcParams['font.size']= 13
neutralDir = "./sourceData/neutral/"
happyDir = "./sourceData/happy/"
fearDir = "./sourceData/fear/"
data2 = []
data =  pd.read_csv("data.csv")
print(len(data["ecg"]))
ecg_signals, info = nk.ecg_process(data["ecg"], sampling_rate=1000)
emg_signals, info = nk.emg_process(data["emg"], sampling_rate=1000)
emgz_signals, info = nk.emg_process(data["emgz"], sampling_rate=1000)
eda_signals, info = nk.eda_process(data["eda"], sampling_rate=1000)


    #print(ecg_signals["ECG_Raw"])
    #pa.pandas2arff(ecg_signals,"test.arff")

    # epochs =  nk.emg_plot(emgz_signals, sampling_rate=1000)
    # epochs =  nk.emg_plot(emg_signals, sampling_rate=1000)
    #plot = nk.ecg_plot(ecg_signals[:10000], sampling_rate=1000)
epochs = nk.ecg_segment(ecg_signals["ECG_Clean"], rpeaks=None, sampling_rate=1000, show=True)
# plt.show()
output = []


# epochs =  nk.emg_plot(emg_signals, sampling_rate=1000)
plt.show()

for key in epochs.keys():
        try:
            data2 += [epochs.get(key)["Signal"].values.tolist()]
        except Exception:
            print(key)
        

    # plt.show()
    

for line in data2:
    print(len(line))
x_train = np.asarray(data2[:len(data2)-10])

x_train.tofile('my_csv.csv', sep=',', format='%s')
#print(np.count_nonzero(np.isnan(x_train)))




# this is the size of our encoded representations
encoding_dim = 32
size = len(data2[0])
np.random.seed(42)  # to ensure the same results


# x_train=x_train+1
# x_train=keras.utils.normalize(x_train)
#media/desvio
autoencoder = Sequential([ 
              Dense(size,input_shape=(size,)), 
              Dense(encoding_dim),
              Dense(size)
])
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train,
            epochs=1000,
            batch_size=64, 
            verbose=2)

out=autoencoder.predict(x_train)
print(out)
x = range(0,len(out[0]))
for cycle in out:
    plt.plot(x,cycle)
plt.show()

