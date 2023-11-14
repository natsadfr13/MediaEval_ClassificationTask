from constants import *
import numpy as np

def prepare_data_for_training(strokeList) -> tuple[np.array, np.array, np.array]:
    """
    Prepare data for training

    Parameters
    ----------
    trickList : list
        List of Trick objects
    
    Returns
    -------
    tuple
        X, y, y_encoded
    
    """
                     
    X = np.array([stroke.data[SELECTED_FEATURES] for stroke in strokeList], dtype=np.float64)
    # see if there are missing values
    X = np.nan_to_num(X)
    y = np.array([stroke.label for stroke in strokeList])
    y_encoded = [STROKE_TO_CLASS[stroke] for stroke in y]
    y_encoded = np.array(y_encoded)

    return X, y, y_encoded


