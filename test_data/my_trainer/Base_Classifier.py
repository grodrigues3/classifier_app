

class Base_Classifier:
    """
    Abstract class for all my web application classifiers
    Specifies the methods that all subclasses should implement
    """
    def __init__(self):
        pass


    def fit(self,X,y):
        pass

    
    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    
