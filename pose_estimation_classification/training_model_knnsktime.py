### Import libraries
import numpy as np
from constants import *

from services.splitTrainTest import splitTrainTest
from services.prepare_data_for_training import prepare_data_for_training
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from services.plot_confusion_matrix import plotConfusionMatrix
from sklearn.model_selection import GridSearchCV

def trainKNN(listStrokesTrain, listStrokesValidation):
    scores = []
        
    X_train, y_train, y_train_encoded = prepare_data_for_training(listStrokesTrain)

    X_test, y_test, y_test_encoded = prepare_data_for_training(listStrokesValidation)

    # Create a KNeighborsTimeSeriesClassifier with DTW distance and 3 neighbors
    # knn = KNeighborsTimeSeriesClassifier(n_neighbors=3, distance='euclidean', weights='distance')
    # knn = KNeighborsTimeSeriesClassifier(n_neighbors=4, distance='squared', weights='distance') #erp, twe

    for n_neighbors in [1, 2, 3, 4, 5, 6, 7]:
        for distance in ['dtw', 'euclidean', 'erp', 'twe', 'squared']:
            for weights in ['uniform', 'distance']:
                knn = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors, distance=distance, weights=weights)
                knn.fit(X_train, y_train_encoded)
                scores.append({'n_neighbors': n_neighbors, 'distance': distance, 
                            'weights': weights, 'score': knn.score(X_test, y_test_encoded)})

        #print(scores)
        # # Find the best scores
    for score in scores:
        if score['score'] == max([score['score'] for score in scores]):
            print(score)

    # # # Fit the classifier to the training data
    import time
    # time1 = time.time()
    # knn.fit(X_train, y_train_encoded)
    # print("Time to fit: ", time.time() - time1)
    # Save the trained classifier to disk
    # knn.save(PATH_TO_MODELS+"knn_squared_10tricks_89%")

    # # Predict the labels for the test set
    time1 = time.time()
    y_pred = knn.predict(X_test)
    print("Time to predict: ", time.time() - time1)
    print(y_pred)
    print(y_test_encoded)
    print("score: ", knn.score(X_test, y_test_encoded))
    plotConfusionMatrix(y_test_encoded, y_pred, listClasses=STROKE_TO_CLASS)
