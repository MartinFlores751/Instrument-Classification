import essentia
import numpy as np
from essentia.standard import *
import os
import re


class FeatureExtractor():
    
    def __init__(self, train_folders=None, test_folders=None):
        self.train_path = os.environ['IRMAS_TRAIN']
        self.test_path = os.environ['IRMAS_TEST']
        self.train_folders = train_folders
        self.test_folders = test_folders
        self.num_classes = len(train_folders)

        self.train_X = None
        self.test_X = None
        self.train_y = None
        self.test_y = None


    def __get_label_from_txt(self, file_path):
        """
        Reads text from file at file_path
        Uses first line as label
        """
        labels = []

        with open(file_path, "r") as file:
            for line in file:
                labels.append(line.strip('\t\n'))

        return labels


        
    def __get_labels_from_name(self, file):
        return re.findall(r"\[([A-Za-z0-9_]+)\]", file)


    def __list_files(self, path):
        return tuple(os.listdir(path))
    

    def __extract_features(self, file, folder):
        full_file_path = folder + file
        
        # NEW
        file_loader = MonoLoader(filename=full_file_path)
        
        file_audio = file_loader()
        
        window = Windowing(type='hann')
        spectrum = Spectrum()
        mfcc = MFCC()
        spec_cont = SpectralContrast()

        pool = essentia.Pool()


        for frame in FrameGenerator(file_audio, frameSize=2048, hopSize=512, startFromZero=True):
            spec = spectrum(window(frame))
            
            # MFCC
            mfcc_bands, mfcc_coeffs = mfcc(spec)

            # Spectral Contrast
            spec_coef, spec_valley = spec_cont(spec)
            
            # Save
            pool.add('lowlevel.mfcc', mfcc_coeffs)
            pool.add('lowlevel.mfcc_bands', mfcc_bands)
            pool.add('lowlevel.spec', spec_coef)

        # OLD
        
        # file_loader = MonoLoader(filename=full_file_path)
        # frameCutter = FrameCutter(frameSize=1024, hopSize=512)
        # w = Windowing(type='hann')

        # spec = Spectrum()
        # specCont = SpectralContrast()
        # mfcc = MFCC()

        # pool = essentia.Pool()

        # file_loader.audio >> frameCutter.signal
        # frameCutter.frame >> w.frame >> spec.frame

        # spec.spectrum >> mfcc.spectrum
        # mfcc.bands >> (pool, 'lowlevel.mel_bands')
        # mfcc.mfcc >> (pool, 'lowlevel.mfcc')

        # essentia.run(file_loader)

        return pool['lowlevel.mfcc'], pool['lowlevel.mfcc_bands'], pool['lowlevel.spec']


    def load_training_data(self):
        """
        Reads trainPath and tainFolders to parse traning files
        """
        data = np.empty((0, 59))
        labels = np.empty((0, self.num_classes))
        for folder in self.train_folders:
            files_in_folder = self.__list_files(self.train_path + folder)
            for file in files_in_folder:
                file_label = self.__get_labels_from_name(file)

                for label in list(file_label):
                    if label + "/" in self.train_folders:
                        continue
                    else:
                        file_label.remove(label)
                
                while len(file_label) < self.num_classes:
                    file_label.append('')

                mfccs, mel_bands, specs = self.__extract_features(file, self.train_path + folder)
                mfccs = np.mean(mfccs, axis=0)
                mel_bands = np.mean(mel_bands, axis=0)
                specs = np.mean(specs, axis=0)

                features = np.hstack([mfccs, mel_bands, specs])
                data = np.vstack([data, features])
                labels = np.vstack((labels, file_label))

        self.train_X = data
        self.train_y = labels
        return data, labels


    def load_testing_data(self):
        """
        Reads testPath and testFolder to parse test folders
        """
        data = np.empty((0, 59))
        labels = np.empty((0, self.num_classes))

        for folder in self.test_folders:
            files_in_folder = self.__list_files(self.test_path + folder)
           
            proper_files = []

            for file in files_in_folder:
                if file.endswith(".txt"):
                    proper_files.append(file[:-4])

            for file in proper_files:
                file_label = self.__get_label_from_txt(self.test_path + folder + file + ".txt")
                isValid = False

                for train in self.train_folders:
                    for label in file_label:
                        if train[:-1] == label:
                            isValid = True
                            break

                if not isValid:
                    continue

                mfccs, bands, specs = self.__extract_features(file + ".wav", self.test_path + folder)
                mfccs = np.mean(mfccs, axis=0)
                bands = np.mean(bands, axis=0)
                specs = np.mean(specs, axis=0)

                for label in list(file_label):
                    if label + "/" in self.train_folders:
                        continue
                    else:
                        file_label.remove(label)
                
                while len(file_label) < 3:
                    file_label.append('')

                features = np.hstack([mfccs, bands, specs])
                data = np.vstack([data, features])
                labels = np.vstack([labels, file_label])

        self.test_X = data
        self.test_y = labels
        return data, labels


    def load_test_train_data(self):
        self.load_training_data()
        self.load_testing_data()
        return self.train_X, self.test_X, self.train_y, self.test_y

