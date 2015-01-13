import pdb

import scipy.sparse as ssp
import numpy as np
import scipy as sp


BASEDIR = "../data/"


class Cleaner:
    def __init__(self):
        pass

    def buildMatrix(self, csvFileName, **kwargs):
        D = 2**15 # 32 kb for now
        for key in kwargs:
            if key == 'D':
                D = kwargs[key]
                print "The number of features allowed will be ", 30
        cols = []
        rows = []
        with open(BASEDIR+csvFileName, 'r') as f:
            for row, line in enumerate(f):
                lab,txt = line.split("\t")
                for i, word in enumerate(txt.split()):
                    cols += [ hash(word)%D]
                    rows += [row] 
                numRows = rows[-1] + 1
                mat = ssp.csr_matrix( (sp.ones(len(rows)), (rows, cols)), shape= (numRows, D), dtype = np.int32)
                if row % 100 == 0:
                    print row, i
            return mat
    
    def stemLine(self, line):
        pass

    def removeStopWords(self, line):
        pass


    

def testCleaner():
    C = Cleaner()
    C.buildMatrix('train.csv')


if __name__ == "__main__":
    testCleaner()
