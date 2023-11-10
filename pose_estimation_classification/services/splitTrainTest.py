import numpy as np

def splitTrainTest(strokes, test_size=0.2, random_state=0):
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
    strokeDict = {}
    for stroke in strokes:
            if stroke.label not in strokeDict:
                    strokeDict[stroke.label] = []
            strokeDict[stroke.label].append(stroke)
    strokeListTrain = []
    strokeListTest = []
    for key in strokeDict:
            if random_state:
                    np.random.seed(random_state)
            np.random.shuffle(strokeDict[key])
            strokeListTrain += strokeDict[key][int(len(strokeDict[key])*test_size):]
            strokeListTest += strokeDict[key][:int(len(strokeDict[key])*test_size)]
    return strokeListTrain, strokeListTest
