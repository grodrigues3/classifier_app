from sklearn.linear_model import SGDClassifier, LogisticRegression
from my_trainer.Preprocessor import Preprocessor as PP
from my_trainer.Logistic_Regression import Logistic_Regression as MyLR
import numpy as np
import pdb, time
train = 'movie_train.tsv'
test = 'movie_test.tsv'

X = PP()
myD = 10**5
train, trainLabs = X.buildMatrix(train, D = myD)
test, testLabs = X.buildMatrix(test, D = myD)
trainLabs = X.convertLabels(trainLabs)
testLabs = X.convertLabels(testLabs)
#alphaVals = [10**i for i in np.linspace(-7,0,7)]
alphaVals = [10**i for i in range(-7,-3)]
print 'Lambda \t\t Sklearn score \t\t sktime \tmy score \t my time'
for i in alphaVals:
    clf = SGDClassifier(alpha= i, loss='log', penalty='l2', n_iter = 5000)
    #clf = LogisticRegression(C = i, penalty='l2')
    clf2 = MyLR()
    s = time.time()
    clf.fit(train, trainLabs)
    skTime = time.time() - s
    #print "Done training with sklearn", time.time() - s
    
    start = time.time()
    clf2.stochastic_fit(train, trainLabs, 1.0/i)
    
    print "{0:.2e}\t {1:.4f} \t\t {2:.2f} \t {3:.4f} \t {4:.2f}".format(i, clf.score(test, testLabs), skTime, clf2.score(test, testLabs), time.time() -s )
    #pdb.set_trace()
    #best C, score 16237.7673919 0.94831673779


