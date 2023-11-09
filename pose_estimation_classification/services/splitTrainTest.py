import numpy as np

def splitTrainTest(tricks, test_size=0.2, random_state=0):
    """
    Split data into training and test set

    Parameters
    ----------
    trickList : list
        List of Trick objects
    test_size : float, optional
        Size of the test set, by default 0.2
    random_state : int, optional
        Random state, by default 0

    Returns
    -------
    tuple
        trickListTrain, trickListTest

    """
    trickDict = {}
    for trick in tricks:
            if trick.label not in trickDict:
                    trickDict[trick.label] = []
            trickDict[trick.label].append(trick)
    trickListTrain = []
    trickListTest = []
    for key in trickDict:
            if random_state:
                    np.random.seed(random_state)
            np.random.shuffle(trickDict[key])
            trickListTrain += trickDict[key][int(len(trickDict[key])*test_size):]
            trickListTest += trickDict[key][:int(len(trickDict[key])*test_size)]
    return trickListTrain, trickListTest
