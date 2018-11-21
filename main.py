import Models.classfier as C

if __name__ == '__main__':
    """Instrument classifier"""
    
    # Options: "knn", "svm", "dtree", "rnn"
    IC = C.InstrumentClassifier("rnn")
