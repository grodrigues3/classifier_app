from my_trainer.Preprocessor import Preprocessor
from my_trainer.Logistic_Regression import Logistic_Regression as LR
from my_trainer.SGDClassifier import SGDClassifier as SGD
from multiprocessing import Process
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import numpy as np
import time, pdb


## GLOBAL VARIABLES
myPreprocessor = Preprocessor()
myMod = None


def getData(fn, numFeats = None, test=False, rep='Hashing Trick'):
    """
    param fn: the filename containing the training data
    param numFeats: how many features to include when building the matrix
    param test: boolean indicating if there is a label or not
    return dataMat: scipy sparse matrix (csr) with the number of rows equal to the number of lines in fn and the number of columns equal to numFeats
    return dataLabels: numerically encoded dataLabels
    """
    dataMat, labels = myPreprocessor.buildMatrix(fn, D = numFeats, test = test)
    if labels == []:
        return dataMat
    dataLabels = myPreprocessor.convertLabels(labels)
    return dataMat, dataLabels



def getBaseRates( fn):
    """
    @param fn: the input file
    """
    return myPreprocessor.getBaseRates(fn)



def fit(modelType, fn, **params):
    global myMod, myPreprocessor
    """
    @return results: return list of dictionaries of values
    e.g: [ {'Training Error': .95, 'Validation Error': .92, 'Number of Features': 10**6}]
    """
    numFolds = 10
    if modelType == "Sklearn Logistic Regression" \
            or modelType == "Garrett's Logistic Regression":

        #Both Sklearn and my implementation take in essentially the same arguments
        try:
            D = int(params['numFeatures'])
        except:
            try:
                D = int(params['uniqueWords']) * 5
            except:
                D = 3*10**4 #30k 
        try:
            regValues = [float(i) for i in params['regValues'].split(',')]
        except:
            regValues =[10** i for i in range(-2,2)]

        try:
            pen = [params['penalty']]
        except:
            pen = ['l2']
        train, trainLabels = getData(fn, numFeats = D)

        if modelType == "Sklearn Logistic Regression":
            param_grid = {'penalty': pen,  'C': regValues}
            clf = GridSearchCV(LogisticRegression(), param_grid, cv = numFolds)
        elif modelType == "Garrett's Logistic Regression":
            toRet = [] 
            bestScore = 0
            for l in regValues:
                clf = LR()
                trainingData, trainLab, test, testLab = train_test_split(train, trainLabels)
                [best_theta, best_cost] = clf.fit(trainingData, trainLab, lamb = l)
                valScore = clf.score(test, testLab)
                if valScore > bestScore:
                    myMod = clf
                    bestScore = valScore
                toRet += [ {'Lambda': l, 'Validation Score': valScore, 'Training Score': clf.score(trainingData, trainLab) , 'Training Loss': best_cost}]
            return D, toRet
              
    elif modelType == "SGD_Classifier":
        try:
            D = int(params['numFeatures'])
        except:
            try:
                D = int(params['uniqueWords']) * 5
            except:
                D = 10**5
        return D, fit_sgd(fn, **params)
    elif modelType == "Neural Network":
        pass
    elif modelType == "Naive Bayes":
        pass
    res = clf.fit(train, trainLabels).grid_scores_
    toRet = []
    for scoreTuple in res:
        someDict ={}
        for gridParam in scoreTuple[0]:
            someDict[gridParam] = scoreTuple[0][gridParam]
        someDict['Mean Validation Score'] = scoreTuple[1]
        someDict['Number of Features'] = D
        #someDict['Fold Scores'] = scoreTuple[2]
        toRet += [someDict]
    myMod = clf.best_estimator_
    return D, toRet
    

def fit_sgd(fn, numFeatures = None, nIters=None, uniqueWords=None, q=None, **kwargs):
    """
    @return results: return list of dictionaries of values
        e.g: [ {'Training Error': .95, 'Validation Error': .92, 'Number of Features': 10**6}]
    """
    print "Training the model..."
    global myMod
    try:
        dList = [int(x) for x in numFeatures.split(",")]
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
        myMod = SGD()
        res += [myMod.fit(fn, numFeatures, nIter= nIters)] #trainScore, valScore, trainLoss
    return res



def train_test_split(dataMat, labels, train_size = .8):
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


def fit_logistic(dataMat, labels, regValues, t_size=.7):
    start = time.time()
    X_train, y_train, X_test, y_test = train_test_split(dataMat, labels, train_size = t_size)
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
            myMod = theMod
        valRes += ["{0:.3f}".format(valScore)]
        print "finished training the model for lambda = ", regParam
        print time.time() - start, "elapsed time"
    print time.time() - start, "total time"
    return zip(trainRes, valRes, lambVals)            


def score(X_test, y_test):
    return myMod.score(X_test, y_test)

def predict(test_fn, D, test=False):
    try:
        return myMod.predict(test_fn)
    except:
        data = getData(test_fn, numFeats = D, test= True)
        predicted_vals = myMod.predict(data)#need to get the data first
        return convertBack(predicted_vals)

def convertBack(labels):
    return myPreprocessor.convertEncoding(labels)

def writeToFile(labels, fn):
    with open("./data/output/" + fn, 'w') as f:
        for l in labels:
            f.write(l + "\n")


