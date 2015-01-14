import numpy as np
from scipy import optimize as op
import scipy.io as sio
import random

import pdb

"""
Tested With scipy version 13.0 and up
Some optimization issues with lower versions
"""
class Logistic_Regression():

    def __init__(self):
        pass
      
    def sigmoid(self,z):
      denom = np.add(1.0, np.exp(-1*z))
      g = np.divide(1.0, denom)
      return g

    def fit(self,X,y,lamb, weights =None):
      class_dict = {y[c]:1 for c in range(len(y))}
      num_classes = len(class_dict)
      if num_classes > 2:
            return self.multiClassFit(X,y,lamb, class_dict, weights)
      else:
          theta_0 = np.zeros(X.shape[1])
          costF = lambda th: self.costFunctionReg(th,X,y,lamb, weights)
          gradF = lambda th: self.costGrad(th,X,y,lamb, weights)
          results = op.fmin_ncg(costF, theta_0, fprime = gradF, maxiter = 100, full_output =1, disp=True)
          #results = op.fmin_bfgs(costF, theta_0, fprime = gradF, maxiter = 400, full_output =1 )
          best_theta = results[0]
          best_cost = results[1]
          self.best_theta = np.array([best_theta])
          return [best_theta, best_cost]
  
    def multiClassFit(self, X,y,lamb, class_dict, weights=None):
        num_classes = len(class_dict)
        print "Fitting the model with " + str(num_classes) + " classes"
        num_features = X.shape[1] #one extra for the bias
        thetas = np.zeros((num_classes, num_features))
        for c in class_dict:
            bin_case = (y==c).astype('float')
            print "The number of classes with y = " + str(c)+ ": ", bin_case.sum()
            best_theta, cost = self.fit(X,bin_case, lamb, weights)
            thetas[c,:] = best_theta
        self.best_theta = thetas
        return [thetas, "No Cost for MultiClass"]
            
            
    def costFunctionReg(self, theta,X,y,lamb, weights=None):
      y = np.array(y).transpose()
      m =  len(y)
      z = X.dot(theta)
      h = self.sigmoid(z)  #X: m*n theta: n*1
      if weights == None:
            weights = np.ones(h.shape)
      first = (weights*np.log(h)).T.dot(-y)
      second = (weights*np.log(1-h)).T.dot(1-y)
      J = 1.0/m* (first - second)
      # Add in regularization
      reg = lamb/(2.0*m) * np.dot(theta[1:],theta[1:])
      J += reg
      return J

    def costGrad(self,theta,X,y,lamb, weights=None):
      
      y = np.array(y).transpose()
      m = len(y)
      z = X.dot(theta) 
      h = self.sigmoid(z).T  #X: m*n theta: n*1
      if weights == None:
            weights = np.ones(h.shape)
      grad = np.zeros(theta.shape)
      #grad =1/m*X'*(h - y) ;
      diff = (h - y) * weights
      grad = 1.0/m * X.T.dot(diff)
      grad[1:] += (lamb*theta[1:])*1.0/m
      return grad
    
    def predict_proba(self,X_test):
        return self.sigmoid( X_test.dot(self.best_theta.T)) 
    
    def predict(self,X_test, threshold=.5):
        if self.best_theta.shape[0] > 1:
            return self.predict_one_vs_all(X_test)
        probs = self.predict_proba(X_test)
        return probs > threshold
        
    def predict_one_vs_all(self,X_test):
        probs = X_test.dot(self.best_theta.T) #n x m by m x C
        return probs.argmax(axis=1)
    
    def score(self, X_test,y_test, verbose = False):
        predictions = self.predict(X_test)
        if verbose:
            print np.column_stack((predictions,y_test))
        correct =  np.equal(predictions,np.array([y_test]).T).sum()
        pdb.set_trace()
        return float(correct)/len(y_test)
        

def test_class_1():
    """
    Test_Class 1
    simple model that relies on the first term to be non zero
    """
    X = np.array([[0, 1, 4],[3, 2, 2], [7 ,0 ,2]])
    y = np.array([0, 1, 1])
    mod = Logistic_Regression()
    print mod.fit(X,y,1)


def test_class_2():
    print "Test Class 2\n"
    my_data = np.genfromtxt('./model_tests/ex2data1.txt', delimiter=',')
    X = my_data[:,0:2]
    y = my_data[:,-1]
    ones = np.zeros((X.shape[0],1)) + 1
    X = np.hstack([ones,X])
    mod = Logistic_Regression()
    theta = np.zeros(X.shape[1])
    [opt_theta, opt_cost]= mod.fit(X,(y==1).astype('int'),0)
    print "Cost", opt_cost
    print "Optimal Theta", opt_theta
    print "\tCost at theta found by fminunc: 0.203506 \n \
            Optimal Theta: -24.932920 0.204407 0.199617" 
    

        
def test_class_3():
    print "Test Class 3\n"
    my_data = np.genfromtxt('./model_tests/ex2data2.txt', delimiter=',')
    X = my_data[:,0:2]
    y = my_data[:,-1]
    ones = np.zeros((X.shape[0],1)) + 1
    X = np.hstack([ones,X])
    mod = Logistic_Regression()
    theta = np.zeros(X.shape[1])
    #print "cost at initial theta", mod.costFunctionReg(theta,X,y,0)
    #print "grad at initial theta", mod.costGrad(theta,X,y,0)
    opt_theta, opt_cost= mod.fit(X,y,0)
    print "Cost", opt_cost
    print "Optimal Theta", opt_theta
    
    print "\tCost at theta found by fminunc: 0.693 \n \
            Optimal Theta: -24.932920 0.204407 0.199617" 
    return mod, opt_theta, opt_cost

def test_class_4():
    print "Test Class 4\n\
        One vs All Classification"
    my_data = sio.loadmat('./model_tests/ex3data1.mat')
    X = my_data['X']
    y = [x[0]%10 for x in my_data['y']]
    ones = np.zeros((X.shape[0],1)) + 1
    X = np.hstack([ones,X])
    mod = Logistic_Regression()
    theta = np.zeros(X.shape[1])
    #print "cost at initial theta", mod.costFunctionReg(theta,X,y,0)
    #print "grad at initial theta", mod.costGrad(theta,X,y,0)
    opt_theta, opt_cost= mod.fit(X,y,0)
    
    print "\tShould achieve > .95 accuracy for 15 iterations of fmin_ncg\n \
            \tCoursera got 94.9% with Gradient Descent"
    print "My score :{0}".format(mod.score(X,y))


if __name__ == "__main__":
    print "-"*50
    #test_class_2()
    print "-"*50
    #test_class_3()
    print "-"*50
    test_class_4()
