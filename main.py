import os
from sklearn.preprocessing import LabelBinarizer
import librosa
import numpy as np
import tensorflow as tf

# Very slooooow!!!!
# Keep it simple!!! Using two instruments for now...
# Warning! Loads 3.4 GB of wav to ram!!!
# There is an error in some variable initialization

trainPath = os.environ['IRMAS_TRAIN']
testPath = os.environ['IRMAS_TEST']
trainFolders = ('cel/', 'cla/', 'flu/')
testFolders = ('part1/', 'part2/', 'part3/')
# , 'gac/', 'gel/', 'org/', 'pia/', 'sax/', 'tru/', 'vio/', 'voi/')


def list_files(path):
    return tuple(os.listdir(path))


def extract_features(file, folder):
    y, sr = librosa.load(dataPath + folder + file, mono=True)
    stft = np.abs(librosa.stft(y))
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,
                       axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_train_files_to_np():
    data = np.empty((0, 193))
    labels = np.empty(0)
    for folder in folders:
        files_in_folder = list_files(trainPath + folder)
        print("Extraing data for the " + folder[:-1] + " instrument.")
        for file in files_in_folder:
            mfccs, chroma, mel, contrast, tonnetz = extract_features(file,
                                                                     folder)
            features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            data = np.vstack([data, features])
            labels = np.append(labels, folder[:-1])
    return data, labels


def parse_test_files_to_np():
    data = np.empty((0, 193))
    labels = np.empty(0)
    return data, labels


def one_hot_encode(labels):
    enc = LabelBinarizer()
    return enc.fit_transform(labels)


def MNN(train_x, train_y, test_x, test_y):
    print("Train X: ", train_x)
    print("Train Y: ", train_y)

    print("\nThe shape of X: ", train_x.shape)
    print("\nThe shape of Y: ", train_y.shape)

    training_epochs = 50
    n_dim = train_x.shape[1]
    n_classes = 3
    n_hidden_units_one = 280
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.0001

    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0,
                                       stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0,
                                       stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

    W_2 = tf.Variable(tf.random_normal(
        [n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0,
                                       stddev=sd))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0,
                                     stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
    y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

    cost_function = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        cost_function)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    y_true, y_pred = None, None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            _, cost = sess.run([optimizer, cost_function],
                               feed_dict={X: train_x, Y: train_y})
            print("Current Cost: ", cost)

        y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: test_x})
        y_true = sess.run(tf.argmax(test_y, 1))

        print("Test accuracy: ", round(session.run(accuracy,
                                                   feed_dict={X: ts_features,
                                                              Y: ts_labels}),
                                       3))


def main():
    print("Reading Files...")
    X, y_temp = parse_train_files_to_np()
    y = one_hot_encode(y_temp)
    print("Y is now: ", y)
    print("X is now: ", X)
    print("Y shape is: ", y.shape)
    print("X shape is: ", X.shape)

    print("Done Reading!!!")
    print("Training MNN...")
    MNN(X, y, None, None)


main()
