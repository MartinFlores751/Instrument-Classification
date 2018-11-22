from DataHandler import data_man as dm
from . import svm, knn, dtree, rnn


class InstrumentClassifier:

    def __init__(self, classifier):
        self.classifier = classifier

        # Run the chosen classifier
        if self.classifier == "svm":
            svm.svm_run()
        elif self.classifier == "knn":
            knn.knn_run()
        elif self.classifier == "dtree":
            dtree.dtree_run()
        elif self.classifier == "rnn":
            rnn.rnn_run()


