from my_trainer.Preprocessor import Preprocessor
from my_trainer.Logistic_Regression import Logistic_Regression as LR
from my_trainer.SGDClassifier import SGDClassifier as SGD
import numpy as np
import time

class Classifier_Model:
    def __init__(self, modelType = 'LR'):
        self.modelType = None
        self.myPreprocessor = Preprocessor()
        self.myMod = None


    def getData(self, fn, myDelimiter = None, numFeats = None, test=False, rep='Hashing Trick'):
        dataMat, labels =  self.myPreprocessor.buildMatrix(fn, D = numFeats, test = test)
        if labels == []:
            return dataMat
        dataLabels = self.myPreprocessor.convertLabels(labels)
        return dataMat, dataLabels



    def getBaseRates(self, fn, delimiter = ","):
        return self.myPreprocessor.getBaseRates(fn, delimiter)

    
    def fit_sgd(self, fn, D = None, nIters=None, val=False):
        if D == None:
            D == 2**20
        if nIters == None:
            nIters = 60
        self.myMod = SGD(D=D)
        return self.myMod.fit(fn, nIter= nIters, validation =val) #trainScore, valScore, trainLoss



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


    def perform_CV(self, modelType, dataMat, labels, regValues, t_size=.7):
        start = time.time()
        X_train, y_train, X_test, y_test = self.train_test_split(dataMat, labels, train_size = t_size)
        trainRes, valRes, lambVals = [], [], []
        bestSoFar = 10**-50
        for regParam in regValues:
            theMod = LR()#instantiate the model
            theMod.fit(X_train, y_train, lamb = regParam) 
            lambVals += [regParam]
            trainRes += ["{0:.3f}".format(theMod.score(X_train,y_train))]
            valScore = theMod.score(X_test,y_test)
            if valScore > bestSoFar:
                bestSoFar = valScore
                self.myMod = theMod
            valRes += ["{0:.3f}".format(valScore)]
            print "finished training the model for lambda = ", regParam
            print time.time() - start, "elapsed time"
        print time.time() - start, "total time"
        return zip(trainRes, valRes, lambVals)            


        

    def score(self, X_test, y_test):
        return self.myMod.score(X_test, y_test)

    def predict(self, X_test, test=False):
        
        fn = self.myMod._get_path(X_test, test=test)
        return self.myMod.predict(fn)

    def convertBack(self, labels):
        return self.myPreprocessor.convertEncoding(labels)

    def writeToFile(self, labels, fn):
        with open("./data/output/" + fn, 'w') as f:
            for l in labels:
                f.write(l + "\n")


