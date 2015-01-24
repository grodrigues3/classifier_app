from multiprocessing import Pool, Process
import classifier_model as CM
filename = 'movie.tsv'
numFeats = '100'
nIters = 10
uniqueWords = 10000
p = Pool(processes = 2)

"TRAINING MY PROCESS"
p2 = Process(target = CM.fit_sgd, args = (filename,  numFeats,  nIters,  uniqueWords))
g = p2.start()
print g, "THIS IS the result"

print "I am going to continue doing other work"
for i in range(100000):
    pass
#this will get called after apply async finishes (maybe can call a view later)
"""
def myFun(x):
    print x
    for key in x[0]:
        print key, x[0][key]

res = CM.fit_sgd(filename,  numFeats,  nIters,  uniqueWords)
print res


res = p.apply_async(CM.fit_sgd, (filename,  numFeats,  nIters,  uniqueWords), callback = myFun)
nIters = 100

def countTo(n):
    for i in range(n):
        print i


res.get()

"""
