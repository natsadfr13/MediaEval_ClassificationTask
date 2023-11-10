# imports
import numpy as np
from constants import *
from services.plot_loss_and_accuracy import plotLossAndAccuracy
from services.plot_confusion_matrix import plotConfusionMatrix
from services.save_model import saveModel
from services.prepare_data_for_training import prepare_data_for_training
from services.splitTrainTest import splitTrainTest
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras import optimizers
from keras.utils import to_categorical

def trainCNN(listStrokesTrain, listStrokesValidation):

    ### Create a validation set
    listTrain, listVal = splitTrainTest(listStrokesTrain, test_size=0.1, random_state=42)

    X_train, y_train, y_train_encoded = prepare_data_for_training(listTrain)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    X_val, y_val, y_val_encoded = prepare_data_for_training(listVal)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

    X_test, y_test, y_test_encoded = prepare_data_for_training(listStrokesValidation)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    nb_classes = len(STROKE_TO_CLASS)
    ### Convert class vectors to binary class matrices
    Y_train = to_categorical(y_train_encoded, nb_classes)
    Y_val = to_categorical(y_val_encoded, nb_classes)
    Y_test = to_categorical(y_test_encoded, nb_classes)

    ### Create a neural network
    ### Set parameters

    temporalCNNmodel = Sequential()
    temporalCNNmodel.add(Conv2D(64, kernel_size=(5, 1), activation='relu'))
    temporalCNNmodel.add(AveragePooling2D(pool_size=(2, 1)))
    temporalCNNmodel.add(Conv2D(32, kernel_size=(4, 1), activation='relu'))
    temporalCNNmodel.add(AveragePooling2D(pool_size=(2, 1)))
    temporalCNNmodel.add(Flatten())
    temporalCNNmodel.add(Dense(nb_classes, activation='softmax'))
    learning_rate = 0.0001
    opt = optimizers.Ftrl(lr=learning_rate)
    temporalCNNmodel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    from keras.callbacks import EarlyStopping
    import time
    time1 = time.time()
    temporalCNNhistory = temporalCNNmodel.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                                            epochs = 200, batch_size = 1, verbose=1,
                                            callbacks=[EarlyStopping(monitor='val_loss', patience=10, 
                                                                    restore_best_weights=True)])
    print("Training time: ", time.time() - time1)
    #saveModel(temporalCNNmodel, "cnn_all_tricks_80")
    #print(temporalCNNmodel.summary())
    #plotLossAndAccuracy(history = temporalCNNhistory)

    ### Plot the confusion matrix
    time1 = time.time()
    y_pred = temporalCNNmodel.predict(X_test)
    print("Prediction time: ", time.time() - time1)
    y_pred = np.argmax(y_pred, axis=1)
    print("score : ", temporalCNNmodel.evaluate(X_test, Y_test, verbose=0)[1])
    plotConfusionMatrix(y_test_encoded, y_pred, STROKE_TO_CLASS)
