import pdb

import scipy.sparse as ssp
import numpy as np
import scipy as sp


BASEDIR = "../data/"


class Cleaner:
    def __init__(self):
        self.mat = None
        self.labs = None
        self.labelDict = {}

    def buildMatrix(self, csvFileName, **kwargs):
        D = 2**15 # 32 kb for now
        printFreq = None
        for key in kwargs:
            if key == 'D':
                D = kwargs[key]
                print "The number of features allowed will be ", D
            if key == 'printFreq':
                printFreq = kwargs[key]


        cols = []
        rows = []
        labels = []
        with open(BASEDIR+csvFileName, 'r') as f:
            for row, line in enumerate(f):
                lab,txt = line.split("\t")
                for i, word in enumerate(txt.split()):
                    cols += [ hash(word)%D]
                    rows += [row] 
                numRows = rows[-1] + 1
                labels += [lab]
                if printFreq and row%printFreq == 0:
                    print row, i
            mat = ssp.csr_matrix( (sp.ones(len(rows)), (rows, cols)), shape= (numRows, D), dtype = np.int32)
            return mat, labels

    def convertLabels(self, labels):
        """ 
        Convert the string representation of labels to a numpy numeric array.
        This simulataneously populates the class attribute labelDict
        """
        counter = 0
        numericLabels = []
        for label in labels:
            if label not in self.labelDict:
                self.labelDict[label] = counter
                counter += 1
            numericLabels += [self.labelDict[label]]
        return np.array(numericLabels)


    
    def stemLine(self, line):
        pass

    def removeStopWords(self, line):
        pass


    



if __name__ == "__main__":
    def testCleaner():
        C = Cleaner()
        C.buildMatrix('train.csv', printFreq = 100)
    testCleaner()
