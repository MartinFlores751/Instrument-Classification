import os
import tensorflow
import librosa

# Warning! Loads 3.4 GB of wav to ram!!!
# Use LibRosa

dataPath = '/home/marflo356/Documents/CSCI4352/Charm/Data/IRMAS/IRMAS-TrainingData/'
folders = ('cel/', 'cla/', 'flu/', 'gac/', 'gel/', 'org/', 'pia/', 'sax/', 'tru/', 'vio/', 'voi/')


def list_files(path):
    return tuple(os.listdir(path))


def main():
    data = []
    for folder in folders:
        filesInFolder = list_files(dataPath + folder)
        data.append([])
        for file in filesInFolder:
            y, sr = librosa.load(dataPath + folder + file, mono=True)
            time = librosa.get_duration(y=y, sr=sr)
            data[len(data)-1].append(tuple(y, sr, time))


main()
