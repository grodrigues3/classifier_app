

import numpy as np
fn = 'movie.tsv'
name,ext = fn.split(".")
t_size = .8
with open(fn, 'r') as f, open(name +"_train." + ext, 'w') as g, \
        open(name + "_test." + ext, 'w') as h, open(name + '_test_labels.'+ext, 'w') as k:
    numLines= 0
    numPos = 0
    numNeg = 0
    for i, line in enumerate(f):
        lab, txt = line.split("\t")
        if np.random.random() < t_size:
                g.write(line)
                if False:
                    if lab == "positive" and numPos < 500:
                        g.write(line)
                        numPos +=1
                    elif lab == "negative" and numNeg < 500:
                        g.write(line)
                        numNeg +=1
        else:
            h.write(line)
            #k.write(lab+"\n")


