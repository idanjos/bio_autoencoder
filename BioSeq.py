import keras
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
    def __init__(self, dir, show=False, emotion="Neutral", virhusID="0") -> None:
        self.id = virhusID
        data2 = []
        self.folder = dir
        self.emotion = emotion
        self.resampled = []
        database = dict()
        li = []
        for file in os.listdir(dir):
            li.append(pd.read_csv(dir+file))
            break
        data = pd.concat(li)
        print(len(data["ecg"]))
        ecg = nk.ecg_simulate(duration=10, heart_rate=70)
        ecg_signals, info = nk.ecg_process(data["ecg"], sampling_rate=1000)
        emg_signals, info = nk.emg_process(data["emg"], sampling_rate=1000)
        emgz_signals, info = nk.emg_process(data["emgz"], sampling_rate=1000)
        eda_signals, info = nk.eda_process(data["eda"], sampling_rate=1000)

        data = [
            nk.ecg_clean(ecg_signals["ECG_Clean"],
                         sampling_rate=1000, method="biosppy"),
            nk.emg_clean(emg_signals["EMG_Clean"], sampling_rate=1000),
            nk.emg_clean(emgz_signals["EMG_Clean"], sampling_rate=1000),
            nk.eda_clean(eda_signals["EDA_Clean"], sampling_rate=1000)
        ]
        dataFrame = pd.DataFrame({
            "ECG": data[0],
            "EMG": data[1],
            "EMGZ": data[2],
            "EDA": data[3]
        })
        
        idk = nk.ecg_clean(
            ecg_signals["ECG_Clean"], sampling_rate=1000, method="biosppy")
        epochs = nk.ecg_segment(
            idk, rpeaks=None, sampling_rate=1000, show=False)

        # epochs =  nk.emg_plot(emg_signals, sampling_rate=1000)
        if show:
            nk.signal_plot(dataFrame, subplots=True)
            plt.show()

        for key in epochs.keys():
            try:
                data2 = [epochs.get(key)["Signal"].values.tolist()]
                key2 = str(len(data2[0]))
                print(key2)
                if key2 not in database.keys():
                    database[key2] = []
                database[key2] += [[epochs.get(key)["Signal"].values.tolist(),
                                   dataFrame["EMG"].values.tolist()[epochs.get(key)["Index"].values[0]:epochs.get(key)[
                                       "Index"].values[-1]],
                                   dataFrame["EMGZ"].values.tolist()[epochs.get(key)["Index"].values[0]:epochs.get(key)[
                                       "Index"].values[-1]],
                                   dataFrame["EDA"].values.tolist()[epochs.get(key)["Index"].values[0]:epochs.get(key)[
                                       "Index"].values[-1]]
                ]]
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
        size = len(self.resampled[0][0])

        encoding_dim = 32

        np.random.seed(42)  # to ensure the same results

        self.autoencoder = Sequential([
            Dense(size, input_shape=(4, size)),
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

    def resampleTo(self, size):
        for segment in self.x_train:
            temp = []
            for array in segment:
                temp += [signal.resample(array, size)]

            self.resampled += [temp]    
        self.x_train = np.asarray(self.resampled)

    def save(self,id):
        self.autoencoder.save(id+".mdl")
    def load(self,id):
        self.autoencoder = keras.models.load_model(id+".mdl")
