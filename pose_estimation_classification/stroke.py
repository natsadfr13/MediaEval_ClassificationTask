import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from constants import *
from scipy.signal import savgol_filter

class Stroke:

    def __init__(self, data:pd.DataFrame, label:str = "Unlabeled", id:str = ""):
        self.data = data
        self.label = label
        self.id = id
        self.predictedLabel = None
        self.predictedLabelProbability = None
        self.isNormalized = False

    def __str__(self):
        return "Stroke " + self.label + ": " + self.id
    
    def plotStroke(self, columns="1_x"):
        self.data.plot(y=columns)

    def normalize(self):
        # normalize signal
        #column = NORMALIZATION_COLUMN

        self.setIndex() 

        self.dimension(MAX_SIZE_STROKE)

        # self.removeNan()
        # for c in SELECTED_FEATURES_PLOT:
        #     self.removePeaks(c, 50)

        #self.minMaxScaling()
        self.isNormalized = True

    def setIndex(self): #Add an index column to the data
        self.data["indexation"] = self.data.index

    def stretch(self, wantedSize):
        # stretch the stroke 
        # Interpolate data to get the desired length
        new_data = {}
        for col in self.data.columns:
            if col != "indexation":
                new_data[col] = np.interp(np.linspace(0, len(self.data[col])-1, wantedSize), np.arange(len(self.data[col])), self.data[col])
        new_data = pd.DataFrame(new_data)
        new_data["indexation"] = new_data.index
        self.data = new_data

    def compress(self, wantedSize):
        # Compress stroke data by removing points
        compression_indices = np.linspace(0, len(self.data) - 1, wantedSize, dtype=int)
        compressed_data = self.data.iloc[compression_indices]

        self.data = pd.DataFrame(compressed_data.reset_index(drop=True))

    def dimension(self, wantedSize):
        if wantedSize > len(self.data):
            self.stretch(wantedSize)
        elif wantedSize < len(self.data):
            self.compress(wantedSize)

    def minMaxScaling(self, columns = None):
        self.data = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(
            self.data), columns=self.data.columns if columns is None else columns)
        
    def removePeaks(self, dataColumn, windowSize):
        self.data[dataColumn] = savgol_filter(self.data[dataColumn], min(len(self.data), windowSize), 4)

    def removeNan(self):
            self.data[self.data == 0.0] = np.nan
            self.data.interpolate(method='linear', inplace=True)