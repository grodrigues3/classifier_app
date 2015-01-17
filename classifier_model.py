from my_trainer.Preprocessor import Preprocessor
from my_trainer.Logistic_Regression import Logistic_Regression as LR
import numpy as np

class Classifier_Model:
    def __init__(self, modelType = 'LR'):
        if modelType == 'LR':
            self.myMod = LR()
        self.myPreprocessor = Preprocessor()

    def getData(self, fn, myDelimiter = ",", test=False):
        dataMat, labels =  self.myPreprocessor.buildMatrix(fn, D = 1000, delimiter = myDelimiter, test = test)
        if labels == []:
            return dataMat
        dataLabels = self.myPreprocessor.convertLabels(labels)
        return dataMat, dataLabels


    def train_test_split(self, dataMat, labels):
        numExamples = dataMat.shape[0]
        allEx = range(numExamples)
        training = sorted(np.random.choice(allEx, int(.7*numExamples), replace=False))
        test = np.setdiff1d(allEx, training)
        X_train, y_train = dataMat[training,:], labels [training]
        X_test, y_test = dataMat[test, :], labels[test]
        return X_train, y_train, X_test, y_test


    def fit(self, X_train, y_train, lamb = 1):
        print "Training the Model"
        self.myMod.fit(X_train, y_train, lamb)

    def score(self, X_test, y_test):
        return self.myMod.score(X_test, y_test)

    def predict(self, X_test):
        return self.myMod.predict(X_test)

    def convertBack(self, labels):
        return self.myPreprocessor.convertEncoding(labels)

    def writeToFile(self, labels, fn):
        with open("./data/output/" + fn, 'w') as f:
            for l in labels:
                f.write(l + "\n")


