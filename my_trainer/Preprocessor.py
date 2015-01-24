import scipy.sparse as ssp
import numpy as np
import scipy as sp
import pdb


class Preprocessor:
    def __init__(self):
        self.mat = None
        self.labs = None
        self.labelDict = {}
        self.backwards_conversion = {}
        
        self.totalTrainingDocs = None
        self.D = None
        self.baseRates = None

    def getDelimiter(self, fn):
        fileExt = fn.rsplit('.', 1)[1].lower()
        delimiter = ","
        if fileExt == "tsv":
            delimiter = "\t"
        return delimiter

    def buildMatrix(self, fn, test=False, **kwargs):


        BASEDIR = "./data/train/"
        if test:
            BASEDIR = "./data/test/"
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
                        self.getBaseRates(fn)
                        D = self.D
                        print D, 'features will be used'
            if key == 'printFreq':
                printFreq = kwargs[key]

        cols = []
        rows = [] 
        labels = []
        delimiter = self.getDelimiter(fn)
        with open(BASEDIR+fn, 'r') as f:
            for row, line in enumerate(f):
                if test:
                    txt = line
                else:
                    try:
                        lab,txt = line.split(delimiter)
                        labels += [lab]
                    except ValueError:
                        raise ValueError( "Check the formatting of your file.  Either a tab (\t) or comma (,) must separate the label from the document")
                for i, word in enumerate(txt.split()):
                    cols += [ hash(word)%D]
                    rows += [row] 
                numRows = rows[-1] + 1
                if printFreq and row%printFreq == 0:
                    print row, i
            mat = ssp.csr_matrix( (sp.ones(len(rows)), (rows, cols)), shape= (numRows, D), dtype = np.int32)
            self.mat = mat
            self.labs = labels
            return mat, labels
    
    def convertLabels(self, labels):
        """ 
        Convert the string representation of labels to a numpy numeric array.
        This simulataneously populates the class attribute labelDict

        param labels: the English representation of the labels
        returns numerically encoded labels as a np array
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
    def getBaseRates(self, fn):
        """
        INPUT:
        #     self: 
        #     fn: the name of the file to be used for training
        #     delimiter: how are the labels separated from the file
        # OUTPUT:
        #     w: updated model
        #     n: updated count
        """
        BASEDIR = "./data/train/"
        delimiter = self.getDelimiter(fn)
        with open(BASEDIR + fn) as f:
            baseRates = {}
            totalDocs = 0
            uniqueWords = {}
            wordCounter = 0
            for line in f:
                lab_txt = line.split(delimiter)
                
                #label counting
                if lab_txt[0] in baseRates:
                    baseRates[lab_txt[0]] +=1
                else:
                    baseRates[lab_txt[0]] = 1
                words = lab_txt[1].split()
                for word in words:
                    if word not in uniqueWords:
                        uniqueWords[word] = wordCounter
                        wordCounter +=1
                totalDocs += 1
        for label in baseRates:
            baseRates[label] = ( baseRates[label], baseRates[label]*1.0/totalDocs) # e.g {'students': [100, .1] } 

        self.totalTrainingDocs = totalDocs
        self.baseRates = baseRates
        self.D = D = len(uniqueWords)
        #self.vocab = uniqueWords

        return baseRates, totalDocs, D


    def stemLine(self, line):
        pass

    def removeStopWords(self, line):
        pass


    



if __name__ == "__main__":
    def testCleaner():
        C = Preprocessor()
        C.buildMatrix('train.csv', printFreq = 100)
    
    testCleaner()
