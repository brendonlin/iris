import numpy as np
from collections import Counter
from sklearn import utils


def balanceSample(X, y, balance=None, keepDistributed=False):
    totalNrow = X.shape[0]
    counter = Counter(y)
    uniqueClass = list(counter.keys())
    minCounterValue = min(counter.values())
    nClass = len(uniqueClass)
    partRow = totalNrow // nClass
    newXList = []
    newyList = []
    for yclass in uniqueClass:
        partX = X[y == yclass]
        nrow = partX.shape[0]
        sampleIndex = utils.resample(range(nrow), n_samples=partRow)
        newXList.append(partX[sampleIndex])
        newyList.extend([yclass] * partRow)
    newX = np.vstack(newXList)
    newY = np.array(newyList)
    return newX, newY
