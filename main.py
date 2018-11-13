import os
from sklearn.preprocessing import LabelBinarizer
import librosa
import numpy as np
import tensorflow as tf

# Very slooooow!!!!
# Keep it simple!!! Using two instruments for now...
# Warning! Loads 3.4 GB of wav to ram!!!

dataPath = os.environ['IRMAS']
folders = ('cel/', 'cla/')

# , 'flu/', 'gac/', 'gel/', 'org/', 'pia/', 'sax/', 'tru/', 'vio/', 'voi/')

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
    data = np.empty((0,193))
    labels = np.empty(0)
    for folder in folders:
        files_in_folder = list_files(dataPath + folder)
        print("Extraing data for the " + folder[:-1] + " instrument.")
        for file in files_in_folder:
            mfccs, chroma, mel, contrast, tonnetz = extract_features(file, folder)
            features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            data = np.vstack([data, features])
            labels = np.append(labels, folder[:-1])
    return data, labels


def one_hot_encode(labels):
    enc = LabelBinarizer(sparse_output=True)
    one_hot_encoded = enc.fit_transform(labels)
    return one_hot_encoded


# THIS IS FOR TESTING!!!
# TODO: Get more methods to train and run
def MNN(train_x, train_y, test_x, test_y):
    print("Train X: ", train_x)
    print("Train Y: ", train_y)
    
    training_epochs = 50
    n_dim = train_x.shape[1]
    n_classes = 2
    n_hidden_units_one = 280 
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.01
    
    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

    W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
    y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

    cost_function = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    y_true, y_pred = None, None
    with tf.Session() as sess:
        tf.global_variables_initializer()
        for epoch in range(training_epochs):            
            _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y})
            print("Current Cost: ", cost)

    
def main():
    print("Reading Files...")
    X, y_temp = parse_files_to_np()
    y = one_hot_encode(y_temp)
    print("Done Reading!!!")
    print("Training MNN...")
    MNN(X, y, None, None)


main()
