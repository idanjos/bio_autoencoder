from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import arff
import pandas2arff as pa
import os
from scipy import signal


class BioSequential():
    def __init__(self, dir, show=False, emotion="Neutral") -> None:
        data2 = []
        self.folder = dir
        self.emotion = emotion
        self.resampled = []
        database = dict()
        li = []
        for file in os.listdir(dir):
            li.append(pd.read_csv(dir+file))
        data = pd.concat(li)
        print(len(data["ecg"]))
        ecg_signals, info = nk.ecg_process(data["ecg"], sampling_rate=1000)
        emg_signals, info = nk.emg_process(data["emg"], sampling_rate=1000)
        emgz_signals, info = nk.emg_process(data["emgz"], sampling_rate=1000)
        eda_signals, info = nk.eda_process(data["eda"], sampling_rate=1000)
        idk = nk.ecg_clean(ecg_signals["ECG_Clean"], sampling_rate=1000, method="biosppy")
        epochs = nk.ecg_segment(
            idk, rpeaks=None, sampling_rate=1000,show=True)
        
        # epochs =  nk.emg_plot(emg_signals, sampling_rate=1000)
        plt.show()

        for key in epochs.keys():
            try:
                data2 = [epochs.get(key)["Signal"].values.tolist()]
                key2 = str(len(data2[0]))
                print(key2)
                if key2 not in database.keys():
                    database[key2] = []
                database[key2] += [epochs.get(key)["Signal"].values.tolist()]
            except Exception as err:
                print(key, err)

        data3 = []
        max = 0
        for key in database.keys():
            data3 = database[key]
        self.x_train = np.asarray(data3[:len(data3)-1])
        
        
        pass

    def fit(self):
        self.autoencoder.fit(self.x_train, self.x_train,
                             epochs=1000,
                             batch_size=64,
                             verbose=2)
        pass

    def __repr__(self) -> str:
        pass

    def initAutoEncoder(self):
        size = len(self.resampled[0])

        encoding_dim = 32

        np.random.seed(42)  # to ensure the same results

        self.autoencoder = Sequential([
            Dense(size, input_shape=(size,)),
            Dense(encoding_dim),
            Dense(size)
        ])
        self.autoencoder.compile(optimizer='adam', loss='mse')
        pass

    def getTrain(self):
        return self.x_train

    def predict(self, array) -> list:
        out = self.autoencoder.predict(array)
        return out

    def resampleTo(self,size):
        for array in self.x_train:
            self.resampled += [signal.resample(array, size)]
        self.x_train = np.asarray(self.resampled)
        

    def save(self):
        self.autoencoder.save(self.folder+self.emotion+".mdl")
