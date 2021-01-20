import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def calBAcc(F, C):
    nRow, nColumn = F.shape
    C = C.astype('int')
    
    NN = KNeighborsClassifier(n_neighbors = 1)
    prediction = []
    # LOO validation
    for i in range(nRow):
        NN.fit(F[[x for x in range(nRow) if x != i]],
               C[[x for x in range(nRow) if x != i]])
        prediction.append(NN.predict(F[[i]]).tolist()[0])
    prediction = np.array(prediction)
    
    BAcc = (np.mean(prediction[np.where(C == 0)] == C[np.where(C == 0)]) +
            np.mean(prediction[np.where(C == 1)] == C[np.where(C == 1)])) / 2
    return BAcc

def McTwo(F, C):
    nRow, nColumn = F.shape
    mBAcc = -1
    selected = set([])
    left = set([x for x in range(nColumn)])
    
    while True:
        BAcc, index = -1, -1
        for x in left:
            tempBAcc = calBAcc(F[:,list(selected) + [x]], C)
            if tempBAcc > BAcc:
                BAcc = tempBAcc
                index = x
        if BAcc > mBAcc:
            mBAcc = BAcc
            selected.add(index)
            left.remove(index)
        else:
            break
    
    return F[:, list(selected)]