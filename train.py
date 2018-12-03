from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score

# from tools.dataset import FeatureExtractor

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


def MNN(train_x, train_y, test_x, test_y, num):
    tf.reset_default_graph()
    
    training_epochs = 20000
    n_dim = train_x.shape[1]
    n_classes = num_classes
    n_hidden_units_one = 280
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.3

    X = tf.placeholder(tf.float32, [None, n_dim], name="X")
    Y = tf.placeholder(tf.float32, [None, n_classes], name="y")

    W1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one]), name="input_weight")
    b1 = tf.Variable(tf.random_normal([n_hidden_units_one]), name="input_bias")
    layer1 = tf.nn.tanh(tf.matmul(X, W1) + b1, name="input_layer")

    W2 = tf.Variable(tf.random_normal(
        [n_hidden_units_one, n_hidden_units_two]), name="hidden_weight")
    b2 = tf.Variable(tf.random_normal([n_hidden_units_two]), name="hidden_bias")
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, W2) + b2, name="hidden_layer")

    W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes]), name="output_weight")
    b = tf.Variable(tf.random_normal([n_classes]), name="output_bias")
    hypothesis = tf.nn.sigmoid(tf.matmul(layer2, W) + b, name="output_layer")

    # Cont approx of hamming loss
    cost_function = tf.reduce_mean(Y*(1-hypothesis) + (1-Y)*hypothesis, name="hamming_loss")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        cost_function)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32, name="predictions")
    intermediate = tf.equal(predicted, Y)
    accuracy = tf.reduce_mean(tf.cast(intermediate, dtype=tf.float32), name="accuracy")

    saver = tf.train.Saver()

    y_true, y_pred = None, None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./graphs', sess.graph)

        for epoch in range(training_epochs):
            _, cost = sess.run([optimizer, cost_function],
                               feed_dict={X: train_x, Y: train_y})

            if epoch % 5000 == 0:
                inter, pred, raw, acc = sess.run(
                            [intermediate, predicted, hypothesis, accuracy],
                            feed_dict={X: test_x,
                                       Y: test_y})
        saver.save(sess, './model-data/MLP-pred-' + str(num))
        print("Accuracy is: ", acc, "%")
        print("Hamming Loss: ", cost)

        return acc, cost, pred


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

    print("Train shape: ", train.shape)
    print("Test shape: ", test.shape)
    
    irmas_all = np.vstack([train, test])
    
    data = pd.DataFrame(data=irmas_all, columns=features)
    data.to_csv("data/gel-pia-sax[MFCC][MFCC_BANDS][SPEC][59].csv", index=False)



def main():
    print("Reading Files...")

    # Load Features
    data = pd.read_csv("data/gel-pia-sax[MFCC][MFCC_BANDS][SPEC][59].csv")

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
    train_y = train_y.astype(float)
    test_y = test_y.astype(float)
    
    # Scale Data
    mms = MinMaxScaler()
    train_X = mms.fit_transform(train_X)
    test_X = mms.transform(test_X)

    # Shuffle the training data just in case...
    train = np.append(train_X, train_y, axis=1)
    np.random.shuffle(train)
    
    train_X = train[:, :-3]
    train_y = train[:, -3:]

    print("Done Processing!!!")
    print("Training MNN...")

    # CrossFold validation, test set only for FINAL evaluation

    accuracys = []
    costs = []
    precissions = []
    recalls = []

    kfolds = KFold(n_splits=3, shuffle=True, random_state=42)
    num = 1
    for train_index, test_index in kfolds.split(train_X, train_y):
        train_X_folds = train_X[train_index]
        train_y_folds = train_y[train_index]
        test_X_folds = train_X[test_index]
        test_y_folds = train_y[test_index]
        
        acc, cost, pred = MNN(train_X_folds, train_y_folds, test_X_folds, test_y_folds, num)
        accuracys.append(acc)
        costs.append(cost)

        pred = pred.astype(float)

        prec1 = precision_score(test_y_folds[:,0], pred[:, 0])
        prec2 = precision_score(test_y_folds[:,1], pred[:, 1])
        prec3 = precision_score(test_y_folds[:,2], pred[:, 2])
        
        precissions.append(np.mean([prec1, prec2, prec3]))

        rec1 = recall_score(test_y_folds[:,0], pred[:, 0])
        rec2 = recall_score(test_y_folds[:,1], pred[:, 1])
        rec3 = recall_score(test_y_folds[:,2], pred[:, 2])

        recalls.append(np.mean([rec1, rec2, rec3]))
        num = num + 1

    print("Accuracys: ", accuracys)
    print("Costs: ", costs)
    print("Precissions: ", precissions)
    print("Recalls: ", recalls)

main()
