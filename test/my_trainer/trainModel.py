
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import scipy as sp
import scipy.sparse as ssp
import traceback
import pdb
from csv import DictReader
import time
from sklearn.metrics import make_scorer, log_loss

start = time.time()

def getData(myReader, batchSize=10**5, D=2**20, needsId = False, start_line = None, num_batches = None):
    rows = []
    cols = []
    y = []
    ids = []
    for j, row in enumerate(myReader):
        for key in row:
            val = row[key]
            if key == 'id':
                if needsId:
                    ids += [val]
                continue#skip the id
            if key == 'click':
                y += [1] if int(row[key]) else [0]
                continue
            else:
                if key == 'hour':
                    for i in range(4):
                        tVal = val[i*2:(i+1)*2]
                        colNo = hash(tVal+str(key))%D
                        rows += [j]
                        cols += [colNo]
                else:
                    colNo = hash(str(val) + str(key))%D
                    cols += [colNo]
                    rows += [j]
        if batchSize:
            if j == batchSize-1:
                break
    if needsId:
        return ids, rows, cols, y
    else:
        return rows, cols, y
        
    
    

def trainSGDModel(fn, batchSize=10**5, D=2**20):
    myMod = SGDClassifier(loss ='log',penalty= 'l2', alpha = 1e-4, warm_start = True)
    avg_y = []
    with open(fn) as f:
       myReader = DictReader(f)
       iterCount = 0
       while True:
            numLines = 40428968    
            rows, cols, y = getData(myReader, batchSize, D)
            avg_y += [ sum(y)*1.0/ len(y)]
            numRows = rows[-1] + 1
            data = ssp.csr_matrix( (sp.ones(len(rows)), (rows, cols)), shape= (numRows, D))
            if iterCount %20 == 0:
                print '\tPerforming a grid search for the optimal parameter...'
                params = { 'loss': ['log'], 'alpha': [10**i for i in range(-10,-4)] }
                logScorer = make_scorer(log_loss, greater_is_better=False)
                clf = GridSearchCV(myMod,params, scoring = logScorer)
                clf.fit(data, y)
                print clf.grid_scores_
                myMod = clf.best_estimator_
                print '\t Determined the best param config: ', clf.best_params_, 'with the best score', clf.best_score_
            print 'Fitting the model on iteration: ', iterCount
            print "\tTotal Lines Read: ", myReader.line_num, ' out of ', numLines
            elapsed = int(time.time() - start)
            print "\tElapsed Time (min, sec): ", elapsed/60, ": ", elapsed%60 
            myMod.partial_fit(data,y)
            iterCount +=1
            if myReader.line_num == numLines:
                break
    """
    print sum(avg_y)*1.0 / len(avg_y), ' is the base rate'
    g = open('batch_base_rates.txt', 'w')
    g.write(str(avg_y))
    g.close()
    """
    return myMod

def trainRF(fn, batchSize=10**3, D= 2**16, start_point = None, num_batches):
    """
    start_point and num_batches are for parallel implementations
    """
    with open(fn) as f:
       myReader = DictReader(f)
       if start_point:
          f.seek(start_point)
       iterCount = 0
       while True:
            myMod = RandomForestClassifier (n_estimators=10, criterion='entropy', max_depth=20 oob_score = True) 
            numLines = 40428968    
            rows, cols, y = getData(myReader, batchSize, D, start_line, num_batches)
            numRows = rows[-1] + 1
            data = ssp.csr_matrix( (sp.ones(len(rows)), (rows, cols)), shape= (numRows, D))
            print 'Fitting the model on iteration: ', iterCount
            myMod.fit(data.todense(),y)
            print "\tTotal Lines Read: ", myReader.line_num, ' out of ', numLines
            elapsed = int(time.time() - start)
            print "\tElapsed Time (min, sec): ", elapsed/60, ": ", elapsed%60 
            yield myMod
            iterCount +=1
            if not myReader.line_num > start_point + numLines/num_batches:
                break
    
if __name__ == "__main__":
    train = './data/new_train.csv'
    val = './data/val.csv'
    """
    rows, cols, y = getData(DictReader(open(val, 'r')),batchSize,  D) 
    numRows = rows[-1]+1
    data = ssp.csr_matrix( (sp.ones(len(rows)), (rows, cols)), shape= (numRows, D)).todense()
    print "Model ", i, "'s accuracy on the validation set", model.score(data, y)
    print "Model ", i, "'s  predicted probability (close to .16 hopefully)", model.predict_proba(data)[:,1].mean()
    """
    try:
        from sklearn.externals import joblib
        batchSize = 10**3
        D = 2**16
        for i, model in enumerate(trainRF(train, batchSize, D )) :
            outFile = './randomForests/RF' + str(i)+ '.txt'
            print "Writing to ", outFile
            joblib.dump(model, outFile)
            
        elapsed = time.time() - start
        print "Elapsed Time (min, sec): ", elapsed/60, ": ", elapsed%60 
    except:
        traceback.print_exc()
        pdb.set_trace()
