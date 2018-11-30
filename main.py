from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from tools.dataset import FeatureExtractor

import numpy as np
import pandas as pd
import tensorflow as tf

np.set_printoptions(threshold=np.inf)

# Possible sets are:
# ('cel/', 'cla/', 'flu/', 'gac/', 'gel/', 'org/', 'pia/', 'sax/', 'tru/', 'vio/', 'voi/')

# Current train size: 2107
# Current test size: 1357

train_folders = ('gel/', 'pia/', 'sax/')
test_folders = ('Part1/', 'Part2/', 'Part3/')
num_classes = len(train_folders)


def one_hot_encode(labels):
    valid_labels = []
    for folder in train_folders:
        valid_labels.append(folder[:-1])

    enc = LabelBinarizer()
    enc.fit(valid_labels)
    labels = enc.transform(labels)

    final_label = np.zeros((1, num_classes))

    for label in labels:
        final_label = np.add(final_label, label)

    return final_label


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
                print("Current Cost: ", cost)
                print("Current Iter: ", epoch)


def make_csv():
    # Set features
    
    features = ["mfcc_1"]
    for i in range(2, 14):
        features.append("mfcc_" + str(i))

    for i in range(1, 41):
        features.append("mel_" + str(i))

    for i in range(1, 4):
        features.append("class_" + str(i))

    # Generate Features
    fe = FeatureExtractor(train_folders=train_folders, test_folders=test_folders)
    train_X, test_X, train_y, test_y = fe.load_test_train_data()

    train = np.hstack([train_X, train_y])
    test = np.hstack([test_X, test_y])

    irmas_all = np.vstack([train, test])
    
    data = pd.DataFrame(data=irmas_all, columns=features)
    data.to_csv("data/gel-pia-sax[MFCC][MEL][53].csv", index=False)



def main():
    print("Reading Files...")

    # Load Features
    data = pd.read_csv("data/gel-pia-sax[MFCC][MEL][53].csv")

    print("Done!\nProcessing files...")
    
    # Split data
    train = data[:2107]
    test = data[2107:]

    train_X = train.drop(["class_1", "class_2", "class_3"], axis=1)
    train_y = train[["class_1", "class_2", "class_3"]]

    test_X = test.drop(["class_1", "class_2", "class_3"], axis=1)
    test_y = test[["class_1", "class_2", "class_3"]]

    # Fill in the empty values
    train_y = train_y.fillna("")
    test_y = test_y.fillna("")

    # DataFram to np.array
    train_y = train_y.values
    test_y = test_y.values

    # Dumb Binary Encoding!!!
    train_y[:] = [one_hot_encode(instance) for instance in train_y]
    test_y[:] = [one_hot_encode(instance) for instance in test_y]
    
    # Scale Data
    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    test_X = ss.transform(test_X)

    # Shuffle the training data just in case...
    train = np.append(train_X, train_y, axis=1)
    np.random.shuffle(train)
    print(train.shape)
    
    trainX = train[:, :-3]
    train_y = train[:, -3:]
    print(train_y.shape)

    print("Done Processing!!!")
    print("Training MNN...")

    MNN(train_X, train_y, test_X, test_y)


main()
