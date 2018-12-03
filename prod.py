from train import one_hot_encode
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score

import pandas as pd
import numpy as np
import tensorflow as tf


def main():
    print("Reading Files...")

    # Load Features
    data = pd.read_csv("data/gel-pia-sax[MFCC][MFCC_BANDS][SPEC][59].csv")

    print("Done!\nProcessing files...")
    
    # Split data
    test = data[2107:]

    test_X = test.drop(["class_1", "class_2", "class_3"], axis=1)
    test_y = test[["class_1", "class_2", "class_3"]]

    # Fill in the empty values
    test_y = test_y.fillna("")

    # DataFram to np.array
    test_y = test_y.values

    # Dumb Binary Encoding!!!
    test_y[:] = [one_hot_encode(instance) for instance in test_y]
    test_y = test_y.astype(float)
    
    # Scale Data
    mms = MinMaxScaler()
    test_X = mms.fit_transform(test_X)

    # Shuffle the training data just in case...
    test = np.append(test_X, test_y, axis=1)
    np.random.shuffle(test)
    
    test_X = test[:, :-3]
    test_y = test[:, -3:]

    print("Done Processing!!!")

    # Run the testing data for final evaluation
    with tf.Session() as sess:
        
        new_save = tf.train.import_meta_graph('./model-data/MLP-2/MLP.meta')
        new_save.restore(sess, tf.train.latest_checkpoint('./model-data/MLP-2/'))

        graph = tf.get_default_graph()
        
        X = graph.get_tensor_by_name("X:0")
        y = graph.get_tensor_by_name("y:0")

        hypothesis = graph.get_tensor_by_name("output_layer:0")
        predicted = graph.get_tensor_by_name("predictions:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")

        hypo, pred, acc = sess.run([hypothesis, predicted, accuracy],
                              feed_dict={X: test_X, y: test_y})

        print("Accuracy: ", acc)

        prec1 = precision_score(test_y[:,0], pred[:, 0])
        prec2 = precision_score(test_y[:,1], pred[:, 1])
        prec3 = precision_score(test_y[:,2], pred[:, 2])
        
        print("Precission: ", np.mean([prec1, prec2, prec3]))

        rec1 = recall_score(test_y[:,0], pred[:, 0])
        rec2 = recall_score(test_y[:,1], pred[:, 1])
        rec3 = recall_score(test_y[:,2], pred[:, 2])

        print("Recall: ", np.mean([rec1, rec2, rec3]))


if __name__ == "__main__":
    main()
