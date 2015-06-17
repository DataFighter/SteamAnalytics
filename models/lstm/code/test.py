ave = dict()
import numpy

save = dict()
save['apple'] = 3
save['pear'] = 7
save['banana'] = 7
save['This'] = 10
save['is'] = 20
save['by'] = 15
save['change']=3
save['haha']=1
save['people']=15
save['long']=10
save['a']=20
save['ab'] = 15
save['use'] = 10
save['done'] = 15
save['great']=3
counts = save.values()
keys = save.keys()
sorted_idx = numpy.argsort(counts)[::-1]
worddict = dict()
for idx, ss in enumerate(sorted_idx):
  worddict[keys[ss]] = idx+2
print worddict
