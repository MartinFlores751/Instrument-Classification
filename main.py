import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

from skmultilearn.adapt import MLkNN
from skmultilearn.dataset import load_dataset
# from skmultilearn.dataset import save_to_arff

import librosa
import essentia
from essentia.streaming import *
import numpy as np
import tensorflow as tf
import re

np.set_printoptions(threshold=np.inf)

trainPath = os.environ['IRMAS_TRAIN']
testPath = os.environ['IRMAS_TEST']
# Possible sets are:
# ('cel/', 'cla/', 'flu/', 'gac/', 'gel/', 'org/', 'pia/', 'sax/', 'tru/', 'vio/', 'voi/')

# Pia, Gel, and Voi have the most training samples, so let's use those
# Replace Voi with Sax if you want pure instruments

trainFolders = ('gel/', 'pia/', 'voi/')
testFolders = ('Part1/', 'Part2/')  #('Part3/')
num_classes = len(trainFolders)


def list_files(path):
    return tuple(os.listdir(path))


def extract_features(file, folder):
    full_file_path = folder + file

    file_loader = MonoLoader(filename=full_file_path)
    frameCutter = FrameCutter(frameSize=1024, hopSize=512)
    w = Windowing(type='hann')

    spec = Spectrum()
    specCont = SpectralContrast()
    mfcc = MFCC()

    pool = essentia.Pool()

    file_loader.audio >> frameCutter.signal
    frameCutter.frame >> w.frame >> spec.frame

    spec.spectrum >> mfcc.spectrum
    mfcc.bands >> (pool, 'lowlevel.mel_bands')
    mfcc.mfcc >> (pool, 'lowlevel.mfcc')

    essentia.run(file_loader)

    return pool['lowlevel.mfcc'], pool['lowlevel.mel_bands']


def get_labels_from_name(file):
    return re.findall(r"\[([A-Za-z0-9_]+)\]", file)


def get_label_from_txt(file_path):
    """
    Reads text from file at file_path
    Uses first line as label
    """
    labels = []

    with open(file_path, "r") as file:
        for line in file:
            labels.append(line.strip('\t\n'))

    return labels


def parse_train_files_to_np():
    """
    Reads trainPath and tainFolders to parse traning files
    """
    data = np.empty((0, 53))
    labels = np.empty((0, num_classes))
    for folder in trainFolders:
        files_in_folder = list_files(trainPath + folder)
        print("Extracting data for the " + folder[:-1] + " instrument.")
        for file in files_in_folder:
            file_label = get_labels_from_name(file)
            file_label = one_hot_encode(file_label)

            mfccs, mel_bands = extract_features(file, trainPath + folder)
            mfccs = np.mean(mfccs, axis=0)
            mel_bands = np.mean(mel_bands, axis=0)

            features = np.hstack([mfccs, mel_bands])
            data = np.vstack([data, features])
            labels = np.vstack((labels, file_label))
    return data, labels


def parse_test_files_to_np():
    """
    Reads testPath and testFolder to parse test folders
    """
    data = np.empty((0, 53))
    labels = np.empty((0, num_classes))

    for folder in testFolders:
        files_in_folder = list_files(testPath + folder)
        print("Extracting data from the " + folder + " folder.")

        proper_files = []

        for file in files_in_folder:
            if file.endswith(".txt"):
                proper_files.append(file[:-4])

        for file in proper_files:
            file_label = get_label_from_txt(testPath + folder + file + ".txt")
            isValid = False

            for train in trainFolders:
                for label in file_label:
                    if train[:-1] == label:
                        isValid = True
                        break

            if not isValid:
                continue

            mfccs, bands = extract_features(file + ".wav", testPath + folder)
            mfccs = np.mean(mfccs, axis=0)
            bands = np.mean(bands, axis=0)

            features = np.hstack([mfccs, bands])
            features = np.hstack([mfccs, bands])
            data = np.vstack([data, features])
            labels = np.vstack([labels, one_hot_encode(file_label)])

    return data, labels


def one_hot_encode(labels):
    valid_labels = []
    for folder in trainFolders:
        valid_labels.append(folder[:-1])

    enc = LabelBinarizer()
    enc.fit(valid_labels)
    pre_labels = enc.transform(labels)

    final_label = np.zeros(num_classes)

    for label in pre_labels:
        for valid_label in trainFolders:
            if np.array_equal(label, enc.transform([valid_label[:-1]]).flatten()):
                final_label = np.add(label, final_label)

    return final_label


def isGood(result):
    num_rows = 0
    num_good = 0.0
    for row in result:
        num_rows += 1
        good = False
        for col in result:
            for elem in col:
                if elem is True:
                    good = True
                    break

        if good is True:
            num_good += 1

    return num_good / num_rows


def MNN(train_x, train_y, test_x, test_y):
    training_epochs = 50000
    n_dim = train_x.shape[1]
    n_classes = num_classes
    n_hidden_units_one = 280
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.000001

    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one]))
    b1 = tf.Variable(tf.random_normal([n_hidden_units_one]))
    layer1 = tf.nn.tanh(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal(
        [n_hidden_units_one, n_hidden_units_two]))
    b2 = tf.Variable(tf.random_normal([n_hidden_units_two]))
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes]))
    b = tf.Variable(tf.random_normal([n_classes]))
    hypothesis = tf.nn.softmax(tf.matmul(layer2, W) + b)

    cost_function = -tf.reduce_sum(Y * tf.log(hypothesis))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        cost_function)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    intermediate = tf.equal(predicted, Y)
    accuracy = tf.reduce_mean(tf.cast(intermediate, dtype=tf.float32))

    y_true, y_pred = None, None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            _, cost = sess.run([optimizer, cost_function],
                               feed_dict={X: train_x, Y: train_y})

            if epoch % 5000 == 0:
                inter, raw, acc = sess.run(
                            [intermediate, hypothesis, accuracy],
                            feed_dict={X: test_x,
                                       Y: test_y})
                print("Raw:\n", raw)
                print("Compare is:\n", inter)
                print("Accuracy is: ", acc, "%")
                print("True Acc is: ", isGood(inter))
                print("Current Cost: ", cost)
                print("Current Iter: ", epoch)


def ML_KNN(train_x, train_y, test_x, test_y):
    knn = MLkNN(k=2)
    knn.fit(train_x, train_y)
    prediction = knn.predict(test_x)
    metrics.hamming_loss(test_y, prediction)


def RNN(train_x, train_y, test_x, test_y):
    num_epochs = 100
    total_series_length = 50000
    truncated_backprop_length = 15
    state_size = 4
    num_classes = 2
    echo_step = 3
    batch_size = 5
    num_batches = total_series_length//batch_size//truncated_backprop_length

    return None


def save_data_arff(X, y, f_name):
    save_to_arff(X, y, label_location='start', save_sparse=True, filename=f_fname)


def load_data_arff(file_path):
    return None


def main():
    print("Reading Files...")

    # Generate Data
    trainX, train_y = parse_train_files_to_np()
    testX, test_y = parse_test_files_to_np()

    # Scale Data
    mms = MinMaxScaler()
    trainX = mms.fit_transform(trainX)
    testX = mms.transform(testX)

    # Shuffle the training data just in case...
    train = np.append(trainX, train_y, axis=1)
    np.random.shuffle(train)
    print(train.shape)
    
    trainX = train[:, :-3]
    train_y = train[:, -3:]
    print(train_y.shape)

    # Save Data
    # save_data_arff(trainX, train_y, "train_data.arff")
    # save_data_arff(testX, test_y, "test_data.arff")

    # Load Data
    # load_data_arff()

    print("Train X:\n", trainX.shape)
    print("Train Y:\n", train_y.shape)

    print("Done Reading!!!")
    print("Training MNN...")

    MNN(trainX, train_y, testX, test_y)
    # ML_KNN(trainX, train_y, testX, test_y)


main()
