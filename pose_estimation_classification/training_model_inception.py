### Import libraries
import numpy as np
from constants import *
from services.plot_confusion_matrix import plotConfusionMatrix
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from services.prepare_data_for_training import prepare_data_for_training

def trainInception(listStrokesTrain, listStrokesValidation):

    X_train, y_train, y_train_encoded = prepare_data_for_training(listStrokesTrain)

    X_validation, y_validation, y_validation_encoded = prepare_data_for_training(listStrokesValidation)

    # Create an InceptionTime classifier
    incep = InceptionTimeClassifier(n_epochs=300, batch_size=4, verbose=True)
    import time
    # Fit the classifier
    time1 = time.time()
    incep.fit(X_train, y_train_encoded)
    print("Time to fit: ", time.time() - time1)
    # Save the classifier
    incep.save(PATH_TO_MODELS+"inception80final")

    # Predict the labels for the test set
    time1 = time.time()
    y_pred = incep.predict(X_validation)
    print("Time to predict: ", time.time() - time1)
    print(y_pred)
    print(y_validation_encoded)
    print("Accuracy: ", np.sum(y_pred == y_validation_encoded)/len(y_validation_encoded))
    plotConfusionMatrix(y_validation_encoded, y_pred, listClasses=STROKE_TO_CLASS)


        