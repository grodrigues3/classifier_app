import pdb

import scipy.sparse as ssp
import numpy as np
import scipy as sp



class Preprocessor:
    def __init__(self):
        self.mat = None
        self.labs = None
        self.labelDict = {}
        self.backwards_conversion = {}
        
        self.totalTrainingDocs = None
        self.D = None
        self.baseRates = None

    def buildMatrix(self, fn, test=False, **kwargs):

        fileExt = fn.rsplit('.', 1)[1].lower()
        delimiter = ","
        for key in kwargs:
            if key == "bias" and not kwargs[key]:
                pass
                #don't add the bias

        if fileExt == "tsv":
            delimiter = "\t"

        BASEDIR= ""
        D = 1000
        printFreq = None
        for key in kwargs:
            if key == 'D':
                try:
                    D = int(kwargs[key])
                except:
                    if self.D:
                        D = self.D
                    else:
                        self.getBaseRates(fn, delimiter = delimiter)
                        D = self.D
            if key == 'printFreq':
                printFreq = kwargs[key]


        cols = []
        rows = []
        labels = []
        with open(BASEDIR+fn, 'r') as f:
            for row, line in enumerate(f):
                if test:
                    txt = line
                else:
                    try:
                        lab,txt = line.split(delimiter)
                        labels += [lab]
                    except ValueError:
                        raise ValueError( "Check the formatting of your file.  Either a tab or comma (,) must separate the label from the document")
                        #probably should add some redirect here to the the select train page
                for i, word in enumerate(txt.split()):
                    cols += [ hash(word)%D]
                    rows += [row] 
                numRows = rows[-1] + 1
                if printFreq and row%printFreq == 0:
                    print row, i
            mat = ssp.csr_matrix( (sp.ones(len(rows)), (rows, cols)), shape= (numRows, D), dtype = np.int32)
            bias = sp.ones((mat.shape[0],1))
            mat = ssp.hstack((bias, mat))
            self.mat = mat
            self.labs = labels
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
                self.backwards_conversion[counter] = label
                counter += 1
            numericLabels += [self.labelDict[label]]
        return np.array(numericLabels)

    def convertEncoding(self, encodedLabels):
        trueLabels = []
        if type(encodedLabels) == type([]):
            for value in encodedLabels:
                trueLabels += [self.backwards_conversion[value]]
        else:
            for i in range(encodedLabels.shape[0]):
                trueLabels += [self.backwards_conversion[encodedLabels[i]]]
        return trueLabels

    def getBaseRates(self, fn, delimiter = "\t"):
        BASEDIR = "./data/train/"
        uniqueWords = {}
        with open(BASEDIR + fn) as f:
            g = {}
            totalDocs = 0
            uniqueWords = 0 
            for line in f:
                lab_txt = line.split(delimiter)
                
                #label counting
                if lab_txt[0] in g:
                    g[lab_txt[0]] +=1
                else:
                    g[lab_txt[0]] = 1
                words = lab_txt[1].split()
                for word in words:
                    if word not in uniqueWords:
                        uniqueWords[word] = 1
                    else:
                        uniqueWords[word] +=1
                totalDocs += 1
        for label in g:
            g[label] = ( g[label], g[label]*1.0/totalDocs) # e.g {'students': [100, .1] } 

        self.totalTrainingDocs = totalDocs
        self.baseRates = g
        self.D = len(uniqueWords)

        return g, totalDocs, len(uniqueWords)




    



if __name__ == "__main__":
    def testCleaner():
        C = Preprocessor()
        C.buildMatrix('train.csv', printFreq = 100)
    
    #testCleaner()
