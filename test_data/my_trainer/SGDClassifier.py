#from Base_Classifier import Base_Classifier
import numpy as np
import scipy as sp



class SGDClassifier:

	def init(self):
		self.weights = None

	def sigmoid(self,z):
		denom = np.add(1.0, np.exp(-1*z))
		g = np.divide(1.0, denom)
		return g
	
	def fit(self, X, y, lamb = 0, **kwargs):
		
		numIter = 10
		learningRate = .1
		for key in kwargs:
			if key == 'numIter':
				numIter = kwargs[key]
			if key == 'learningRate':
				learningRate = kwargs[key]
		
		if len(y.shape) > 1:
			bigDim = np.argmax(y.shape)
			if bigDim == 1:
				y = y[0,:]
			else:
				y = y[:,0]


		numEx = X.shape[0]
		numFeatures = X.shape[1]

		weights = np.random.random((numFeatures+1,)) #add one extra for the bias
		bias = sp.ones((numEx,1))
		X = sp.hstack( (bias, X))  #add the bias term to all examples


		for iter in range(numIter):
			for i in range(numEx):
				curEx = X[i,:]
				prob = self.sigmoid(curEx.dot(weights))
				weights -= learningRate*(prob-y[i])*curEx + lamb*weights
		
		self.weights = weights

		cost = self.costFunctionReg(weights, X,y,0)
		return weights, cost
    	

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


	def fit_ftrl(self):
		pass

	def score(self, X, y):
		pass

	def predict(self, X):
		pass


def test_class_2():
    print "Test Class 2\n"
    my_data = np.genfromtxt('./model_tests/ex2data1.txt', delimiter=',')
    X = my_data[:,0:2]
    y = my_data[:,-1]
    mod = SGDClassifier()
    theta = np.zeros(X.shape[1])
    [opt_theta, opt_cost]= mod.fit(X,(y==1))
    print "Cost", opt_cost
    print "Optimal Theta", opt_theta
    print "\tCost at theta found by fminunc: 0.203506 \n \
            Optimal Theta: -24.932920 0.204407 0.199617"


if __name__ == "__main__":
	test_class_2()