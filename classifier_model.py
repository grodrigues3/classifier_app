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

    
    def getData(self, fn, numFeats = None, test=False, rep='Hashing Trick'):
        """
        param fn: the filename containing the training data
        param numFeats: how many features to include when building the matrix
        param test: boolean indicating if there is a label or not
        return dataMat: scipy sparse matrix (csr) with the number of rows equal to the number of lines in fn and the number of columns equal to numFeats
        return dataLabels: numerically encoded dataLabels
        """
        dataMat, labels =  self.myPreprocessor.buildMatrix(fn, D = numFeats, test = test)
        if labels == []:
            return dataMat
        dataLabels = self.myPreprocessor.convertLabels(labels)
        return dataMat, dataLabels


    
    def getBaseRates(self, fn):
        """
        @param fn: the input file
        """
        return self.myPreprocessor.getBaseRates(fn)

   

    def fit(self, modelType, **params):
        """
        @return results: return list of dictionaries of values
            e.g: [ {'Training Error': .95, 'Validation Error': .92, 'Number of Features': 10**6}]
        """
        if modelType == "Logistic Regression":
            pass
        elif modelType == "SGDClassifier":
            pass
        elif modelType == "Neural Network":
            pass
        elif modelType == "Naive Bayes":
            pass
        

    def fit_sgd(self, fn, dList = None, nIters=None, uniqueWords=None):
        """
        @return results: return list of dictionaries of values
            e.g: [ {'Training Error': .95, 'Validation Error': .92, 'Number of Features': 10**6}]
        """
        print "Training the model..."
        try:
            dList = [int(x) for x in dList.split(",")]
        except:
            if uniqueWords:
                dList = [int(uniqueWords)*5]
            else:
                dList = [2**20]
        try:
            nIters = int(nIters)
        except:
            nIters = 100
        res = []
        for numFeatures in dList:
            self.myMod = SGD()
            res += [self.myMod.fit(fn, numFeatures, nIter= nIters)] #trainScore, valScore, trainLoss
        return res



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


