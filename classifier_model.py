from my_trainer.Preprocessor import Preprocessor
from my_trainer.Logistic_Regression import Logistic_Regression as LR
import numpy as np

class Classifier_Model:
    def __init__(self, modelType = 'LR'):
        models = {'LR': LR, 'SGD': 'ToDo', 'NB': 'Too be implemented', 'NN': 'ToDo'}
        self.modelType = models[modelType] #still need to instantiate in fit
        self.myPreprocessor = Preprocessor()

    def getData(self, fn, myDelimiter = ",", test=False):
        dataMat, labels =  self.myPreprocessor.buildMatrix(fn, D = 500, delimiter = myDelimiter, test = test)
        if labels == []:
            return dataMat
        dataLabels = self.myPreprocessor.convertLabels(labels)
        return dataMat, dataLabels


    def train_test_split(self, dataMat, labels, train_size = .7):
        """
        For cross validation purposes.  Split the 
        @param dataMat: numpy array, matrix or scipy sparse matrix 
        @param labels: the training labels
        @param train_size: the percentage of the dataset used for training
        """
        numExamples = dataMat.shape[0]
        allEx = range(numExamples)
        training = sorted(np.random.choice(allEx, int(train_size*numExamples), replace=False))
        test = np.setdiff1d(allEx, training)
        X_train, y_train = dataMat[training,:], labels [training]
        X_test, y_test = dataMat[test, :], labels[test]
        return X_train, y_train, X_test, y_test


    def perform_CV(self, dataMat, labels, regValues, t_size=.7):
        X_train, y_train, X_test, y_test = self.train_test_split(dataMat, labels, train_size = t_size)
        trainRes, valRes, lambVals = [], [], []
        bestSoFar = 10**-50
        for regParam in regValues:
            theMod = self.modelType()#instantiate the model
            theMod.fit(X_train, y_train, lamb = regParam) 
            lambVals += [regParam]
            trainRes += ["{0:.3f}".format(theMod.score(X_train,y_train))]
            valScore = theMod.score(X_test,y_test)
            if valScore > bestSoFar:
                bestSoFar = valScore
                self.myMod = theMod
            valRes += ["{0:.3f}".format(valScore)]
        return zip(trainRes, valRes, lambVals)            


        

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


