'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt


# parameters #################################################################

train = 'movie_train.tsv'  # path to training file
test = 'movie_test.tsv'  # path to testing file

D = 2 ** 20   # number of weights use for learning
alpha = .1    # learning rate for sgd optimization


# function definitions #######################################################

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)


# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(txt_line, D):
    x = [0]  # 0 is the index of the bias term
    for word in txt_line.split():
        index = hash(word) % D  
        x.append(index)
    return x  # x contains indices of features that have a value of 1


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
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
def update_w(w, n, x, p, y):
    for i in x:
        # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        w[i] -= (p - y) * alpha / (sqrt(n[i]) + 1.)
        n[i] += 1.

    return w, n


# training and testing #######################################################

# initialize our model
w = [0.] * D  # weights
n = [0.] * D  # number of times we've encountered a feature

# start training a logistic regression model using on pass sgd
loss = 0.
labelDict = {}
reverseDict = {}
labelCounter = 0
for t, row in enumerate(open(train)):
    lab, txt = row.split("\t")
    if lab in labelDict:
        y = labelDict[lab]
    else:
        y = labelDict[lab] = labelCounter
        reverseDict[labelCounter] = lab
        labelCounter += 1

    # main training procedure
    # step 1, get the hashed features
    x = get_x(txt, D)

    # step 2, get prediction
    p = get_p(x, w)

    # for progress validation, useless for learning our model
    loss += logloss(p, y)
    if t % 1000000 == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent logloss: %f' % (
            datetime.now(), t, loss/t))

    # step 3, update model with answer
    w, n = update_w(w, n, x, p, y)

# testing (build kaggle's submission file)
with open('movie_predict.csv', 'w') as submission:
    submission.write('Id,Predicted\n')
    for t,line in enumerate(open(test)):
        #txt = line  #should only be one  line anyway
        x = get_x(txt, D)
        p = get_p(x, w)
        predicted_label = reverseDict[int(p > .5)]
        submission.write('%f\n, %s' % (p, predicted_label))
