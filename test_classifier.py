import pdb
import numpy as np
from my_trainer.Preprocessor import Preprocessor
from my_trainer.Logistic_Regression import Logistic_Regression as LR


def getData(fn):
    C = Preprocessor(fn)
    dataMat, labels =  C.buildMatrix( D = 1000, delimiter= "\t")
    dataLabels = C.convertLabels(labels) #convert to numeric representation
    return dataMat, dataLabels

def trainModel(X,y, lamb = 1):
    numExamples = X.shape[0]
    allEx = range(numExamples)
    training = sorted(np.random.choice(allEx, int(.7*numExamples), replace=False))
    test = np.setdiff1d(allEx, training)
    X_train, y_train = X[training,:], y[training]
    X_test, y_test = X[test, :], y[test]
    myMod = LR()
    weights, cost = myMod.fit(X_train,y_train,lamb)
    print myMod.score(X_test, y_test)


if __name__ == "__main__":
    X,y = getData('movie.tsv')
    trainModel(X,y)
