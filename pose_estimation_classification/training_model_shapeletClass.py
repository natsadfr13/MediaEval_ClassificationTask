### Import libraries
from constants import *
from services.plot_confusion_matrix import plotConfusionMatrix
from services.splitTrainTest import splitTrainTest
from services.prepare_data_for_training import prepare_data_for_training
from sktime.classification.shapelet_based import ShapeletTransformClassifier

def trainShapelet(listStrokesTrain, listStrokesValidation):
    
    X_train, y_train, y_train_encoded = prepare_data_for_training(listStrokesTrain)

    X_test, y_test, y_test_encoded = prepare_data_for_training(listStrokesValidation)
    
    # Create a ShapeletTransformClassifier
    shapelet = ShapeletTransformClassifier(n_shapelet_samples=1000, batch_size=100)
    import time
    time1 = time.time()
    shapelet.fit(X_train, y_train_encoded)
    print("Time to fit: ", time.time() - time1)

    time1 = time.time()
    y_pred = shapelet.predict(X_test)
    print("Time to predict: ", time.time() - time1)

    print(y_pred)
    print(y_test_encoded)
    print(shapelet.score(X_test, y_test_encoded))
    plotConfusionMatrix(y_test_encoded, y_pred, listClasses=STROKE_TO_CLASS)

    shapelet.save(PATH_TO_MODELS+"shapelet_80")



        