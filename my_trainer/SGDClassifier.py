from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from string import maketrans, punctuation
from random import random
from os import remove

# parameters #################################################################



# function definitions #######################################################
class SGDClassifier:

    def __init__(self, D= 10**6, alpha = 1):
        self.labelDict = {}
        self.reverseDict = {}
        self.w = None

        self.D = D  # number of weights use for learning
        self.alpha = 10    # learning rate for sgd optimization

    # Bounded logloss
    # INPUT:
    #     p: our prediction
    #     y: real answer
    # OUTPUT
    #     logarithmic loss of p given y
    def logloss(self, p, y):
        p = max(min(p, 1. - 10e-12), 10e-12)
        return -log(p) if y == 1. else -log(1. - p)


    # B. Apply hash trick of the original csv row
    # for simplicity, we treat both integer and categorical features as categorical
    # INPUT:
    #     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
    #     D: the max index that we can hash to
    # OUTPUT:
    #     x: a list of indices that its value is 1
    def get_x(self, txt_line, D):
        x = [0]  # 0 is the index of the bias term
        stripped = txt_line.translate(maketrans("",""), punctuation)
        for word in stripped.split():
            index = hash(word) % D  
            x.append(index)
        return x  # x contains indices of features that have a value of 1


    # C. Get probability estimation on x
    # INPUT:
    #     x: features
    #     w: weights
    # OUTPUT:
    #     probability of p(y = 1 | x; w)
    def get_p(self, x, w):
        wTx = 0.
        for i in x:  # do wTx
            wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
        return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


    # D. Update given model
    # INPUT:
    #     w: weights
    #     n: a counter that counts the number of times we encounter a feature
    #        this is used for adaptive learning rate
    #     x: feature
    #     p: prediction of our model
    #     y: answer
    # OUTPUT:
    #     w: updated model
    #     n: updated count
    def update_w(self, w, n, x, p, y):
        for i in x:
            # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
            # (p - y) * x[i] is the current gradient
            # note that in our case, if i in x then x[i] = 1
            alpha = self.alpha
            learningRate  =  alpha / (sqrt(n[i]) + 1.)
            w[i] -= (p - y) * learningRate 
            n[i] += 1.

        return w, n


    def create_split(self, in_file, t_size = .8):
        import os
        print os.getcwd()
        fn = in_file.rsplit('.', 1)[0].lower()
        ext = in_file.rsplit('.', 1)[1].lower()
        trainFn = self._get_path(fn+"_train." + ext)
        valFn = self._get_path(fn+"_val."+ext)
        with open(trainFn, 'w') as f, open(valFn, 'w') as g:
            for row in open(self._get_path(in_file), 'r'):
                if random() < t_size:
                    f.write(row)
                else:
                    g.write(row)

        return trainFn, valFn



    #appends the training or file directory to the path
    def _get_path(self, fn, test = False):
        BASEDIR = "./data/train/"
        if test:
            BASEDIR = "./data/test/"
        return BASEDIR + fn


     # D. Update given model
    # INPUT:
    #     self: 
    #     train: the name of hte file to be used for training
    #     nIter: how many iterations to complete
    #     validation: bool - divide the training set in to train/val sets
    # OUTPUT:
    #     w: updated model
    #     n: updated count
    def fit(self, training_file, nIter = 400, validation = True, **kwargs):
        D = self.D
        # initialize our model
        w = [0.] * D  # weights
        n = [0.] * D  # number of times we've encountered a feature


        # start training a logistic regression model using on pass sgd
        loss = 0.
        if validation:
            training_file, validation_file = self.create_split(training_file, .9)

        labelCounter = 0
        total = 0
        for j in range(nIter):
            for t, row in enumerate(open(training_file)):
                lab, txt = row.split("\t")
                if lab in self.labelDict:
                    y = self.labelDict[lab]
                else:
                    y = self.labelDict[lab] = labelCounter
                    self.reverseDict[labelCounter] = lab
                    labelCounter += 1
                # main training procedure
                # step 1, get the hashed features
                x = self.get_x(txt, D)
                # step 2, get prediction
                p = self.get_p(x, w)

                loss += self.logloss(p, y)
                w, n = self.update_w(w, n, x, p, y)
                total += 1
            print('%s\tencountered: %d\tcurrent logloss: %f\titeration %d' % (
                    datetime.now(), total, loss/total, j))
            #pdb.set_trace()
            if validation:
                self.w = w
                print "Validation Score:", self.score(validation_file)

        print t, "Training Examples Used"
        self.w = w

        trainScore = self.score(training_file)
        if validation:
            valScore = self.score(validation_file)
            
            try:
                #remove the files
                remove(validation_file) 
                remove(training_file)
            except:
                print validation_file
                print training_file
        else:
            valScore = "No Validation Performed"
        trainLoss = loss/total

        return trainScore, valScore, trainLoss

    def score(self, in_file):
        predicted_labels = self.predict(in_file)
        correct = 0
        for i, line in enumerate(open(in_file)):
            trueLabel = line.split("\t")[0]
            if predicted_labels[i] == trueLabel:
                correct += 1
        return correct *1./ (i+1)


    def predict_proba(self, test_fn):
        correct = 0
        all_ps = []
        w = self.w
        D = self.D
        for t,line in enumerate(open(test_fn)):
            check = line.split("\t")
            if len(check) == 1:
                #txt = line  #should only be one  line anyway
                x = self.get_x(check[0], D)
            elif len(check) == 2:
                x = self.get_x(check[1], D)
            p = self.get_p(x, w)
            all_ps += [p]
        return all_ps

    def predict(self, test_fn):
        self.probs = self.predict_proba(test_fn)
        predicted_labels = []
        for p in self.probs:
            predicted_labels += [self.reverseDict[int(p > .5)]]

        return predicted_labels 

    def write_output(self, outfile, labels):
        correct = 0
        with open(outfile, 'w') as submission, open('movie_test_labels.tsv', 'r') as f:
            for i, predicted_label in enumerate(labels):
                p = self.probs[i]
                trueLabel = f.readline().rstrip("\n")
                if predicted_label == trueLabel:
                    correct += 1
                toWrite = "{0:.2f}, {1}\n".format(p, predicted_label)
                submission.write(toWrite)
        print "Final Score: {0}".format(correct*1./ (i+1))

if __name__ == "__main__":
    sgd = SGDClassifier()
    print sgd.fit('movie.tsv')
    #predicted = sgd.predict('movie_test.tsv')
    #sgd.write_output('my_output.csv', predicted)
    #print t
    
