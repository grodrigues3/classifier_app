import pdb
  
with open('train.csv') as f:
    g = {}
    for line in f:
        lab_txt = line.split("\t")
        if lab_txt[0] in g:
            g[lab_txt[0]] +=1
        else:
            g[lab_txt[0]] = 1

print g

#2803
#{'project': 336, 'course': 620, 'student': 1097, 'faculty': 750}}
