import os
from sklearn.preprocessing import LabelBinarizer
import librosa
import essentia
from essentia.streaming import *
import numpy as np
import tensorflow as tf


trainPath = os.environ['IRMAS_TRAIN']
testPath = os.environ['IRMAS_TEST']
trainFolders = ('cel/', 'cla/') # 'flu/', 'gac/', 'gel/', 'org/', 'pia/', 'sax/', 'tru/', 'vio/', 'voi/')
testFolders = ('tmp/', 'tmp2/') # ()'Part3/')


def list_files(path):
    return tuple(os.listdir(path))


def extract_features(file, folder):
    full_file_path = folder + file

    file_loader = MonoLoader(filename=full_file_path)
    frameCutter = FrameCutter(frameSize=1024, hopSize=512)
    w = Windowing(type = 'hann')

    spec = Spectrum()
    specCont = SpectralContrast()
    mfcc = MFCC()

    pool = essentia.Pool()

    file_loader.audio >> frameCutter.signal
    frameCutter.frame >> w.frame >> spec.frame
    spec.spectrum >> mfcc.spectrum

    mfcc.bands >> (pool, 'lowlevel.bands')
    mfcc.mfcc >> (pool, 'lowlevel.mfcc')

    essentia.run(file_loader)

    return pool['lowlevel.mfcc'], pool['lowlevel.bands']


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
    data = np.empty((0, 13780))
    labels = np.empty((0, 11))
    for folder in trainFolders:
        files_in_folder = list_files(trainPath + folder)
        print("Extracting data for the " + folder[:-1] + " instrument.")
        for file in files_in_folder:
            file_label = []
            file_label.append(folder[:-1])
            file_label = (one_hot_encode(file_label))
            mfccs, bands = extract_features(file, trainPath + folder)
            mfccs = mfccs.flatten()
            bands = bands.flatten()
            features = np.hstack([mfccs, bands])
            data = np.vstack([data, features])
            labels = np.vstack((labels, file_label))
    return data, labels


def parse_test_files_to_np():
    """
    Reads testPath and testFolder to parse test folders
    """
    data = np.empty((0, 13780))
    labels = np.empty((0, 11))

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

            mfccs = mfccs[:260][:]
            bands = bands[:260][:]

            mfccs = mfccs.flatten()
            bands = bands.flatten()

            features = np.hstack([mfccs, bands])
            data = np.vstack([data, features])
            labels = np.vstack([labels, one_hot_encode(file_label)])

    return data, labels


def one_hot_encode(labels):
    enc = LabelBinarizer()
    enc.fit(
    ['cel', 'cla', 'flu', 'gac', 'gel', 'org',
    'pia', 'sax', 'tru', 'vio', 'voi'])
    pre_labels = enc.transform(labels)

    final_label = pre_labels[0]

    for label in pre_labels[1:]:
        final_label = np.add(label, final_label)

    return final_label


def MNN(train_x, train_y, test_x, test_y):
    training_epochs = 5000
    n_dim = train_x.shape[1]
    n_classes = 11
    n_hidden_units_one = 280
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.0001

    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0,
                                       stddev=sd))
    b1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0,
                                       stddev=sd))
    layer1 = tf.nn.tanh(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.random_normal(
        [n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
    b2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0,
                                       stddev=sd))
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0,
                                     stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
    hypothesis = tf.nn.softmax(tf.matmul(layer2, W) + b)

    cost_function = -tf.reduce_sum(Y * tf.log(hypothesis))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        cost_function)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    y_true, y_pred = None, None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            _, cost = sess.run([optimizer, cost_function],
                                feed_dict={X: train_x, Y: train_y})

            if epoch % 500 == 0:
                cost, pred, acc = sess.run([cost_function, predicted, accuracy],
                            feed_dict={X: test_x,
                                       Y: test_y})
                print("Predicted is: ", pred)
                print("Acutual is: ", test_y)
                print("Accuracy is: ", acc)
                print("Current Cost: ", cost)


def main():
    print("Reading Files...")

    trainX, train_y = parse_train_files_to_np()
    testX, test_y = parse_test_files_to_np()

    print(train_y)
    print(test_y)

    print("Done Reading!!!")
    print("Training MNN...")

    MNN(trainX, train_y, testX, test_y)


main()
