import os
from sklearn.preprocessing import OneHotEncoder
import librosa
import numpy as np

# Warning! Loads 3.4 GB of wav to ram!!!

dataPath = os.environ['IRMAS']
folders = ('cel/', 'cla/', 'flu/', 'gac/', 'gel/', 'org/', 'pia/', 'sax/', 'tru/', 'vio/', 'voi/')


def list_files(path):
    return tuple(os.listdir(path))


def extract_features(file, folder):
    y, sr = librosa.load(dataPath + folder + file, mono=True)
    stft = np.abs(librosa.stft(y))
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_files_to_np():
    data = []
    for folder in folders:
        files_in_folder = list_files(dataPath + folder)
        data.append([])
        for file in files_in_folder:
            features = extract_features(file, folder)
            data[len(data) - 1].append(features)
    return data


def folders_to_one_hot():
    yLabels = []
    for folder in folders:
        yLabels.append(folder[:-1])
    # TODO: Use sklearn onehot to convert labels!


def main():
    data = parse_files_to_np()


print(type(dataPath))
main()
