import numpy as np
from scipy import optimize as op
import scipy.io as sio
import random
from Base_Classifier import Base_Classifier
import pdb

"""
Tested With scipy version 13.0 and up
Some optimization issues with lower versions
"""
class Logistic_Regression(Base_Classifier):
    """Logistic Regression Implementation using numpy and scipy.  Training is completed with the scipy optimization library and newton-conjugate gradient method"""
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
          results = op.minimize( fun = costF, x0 = theta_0, method= 'L-BFGS-B', jac = gradF) #, maxiter = 10, full_output =2, disp=True)
          
          #results = op.fmin_ncg( f = costF, x0 = theta_0, fprime = gradF, maxiter = 10, full_output =2, disp=True)
          #results = op.fmin_bfgs(costF, theta_0, fprime = gradF, maxiter = 400, full_output =1 )
          #best_theta = results[0]
          #best_cost = results[1]
          best_theta = results['x']
          best_cost = results['fun']
          #pdb.set_trace()
          self.best_theta = np.array([best_theta])
          return [best_theta, best_cost]
  
    def multiClassFit(self, X,y,lamb, class_dict, weights=None):
        num_classes = len(class_dict)
        print "Fitting the model with " + str(num_classes) + " classes"
        num_features = X.shape[1]
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
        return probs.argmax(axis=1)
    
        
    def predict_one_vs_all(self,X_test):
        probs = X_test.dot(self.best_theta.T) #n x m by m x C
        return probs.argmax(axis=1)
    
    def score(self, X_test,y_test, verbose = False):
        predictions = self.predict(X_test)
        if len(predictions.shape) > len(y_test.shape):
            y_test = np.array([y_test]).T
        if verbose:
            print np.column_stack((predictions,y_test))
        correct =  (predictions == y_test).sum()
        return float(correct)/len(y_test)
        


