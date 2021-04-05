import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.externals
from sklearn.externals import joblib
from keras.models                   import Sequential
from keras.layers                   import Dense 
import keras

df = pd.read_csv("data.csv")
# Plot it
# df.plot()

bio = nk.bio_process(ecg=df["ecg"], emg=df["emg"], eda=df["eda"], sampling_rate=1000)

# nk.z_score(bio["df"]).plot()
pd.DataFrame(bio["ECG"]["Cardiac_Cycles"]).plot(legend=False)  #
# plt.show() 
x_train = np.asarray(bio["ECG"]["Cardiac_Cycles"])

# for cycle in x_train:
#     plt.plot(x,cycle)
# plt.show()

x_train.tofile('my_csv.csv', sep=',', format='%s')

encoding_dim = 32
size = len(x_train[0])
np.random.seed(42)  # to ensure the same results

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

print("end")