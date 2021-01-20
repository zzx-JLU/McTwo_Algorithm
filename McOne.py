import numpy as np
from minepy import MINE

def mic(x, y):
    mine = MINE()
    mine.compute_score(x, y)
    return mine.mic()

def McOne(F, C, r):
    nRow, nColumn = F.shape
    micFC = [-1 for _ in range(nColumn)] # store MIC between all features and class
    Subset = [-1 for _ in range(nColumn)] # store subset by the feature ID
    numSubset = 0 # [0, numSubset - 1] contains the selected features
    
    for i in range(nColumn):
        micFC[i] = mic(F[:, i], C)
        if micFC[i] >= r:
            Subset[numSubset] = i
            numSubset += 1
    
    Subset = Subset[:numSubset]
    Subset.sort(key = lambda x: micFC[x], reverse = True)
    
    for e in range(numSubset):
        q = e + 1
        while q < numSubset:
            if mic(F[:, Subset[e]], F[:, Subset[q]]) >= micFC[Subset[q]]:
                for i in range(q, numSubset - 1):
                    Subset[i] = Subset[i + 1]
                numSubset -= 1
            else:
                q += 1
    FReduce = F[:, np.array(Subset)[:numSubset]]
    
    return FReduce