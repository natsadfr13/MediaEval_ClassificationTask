import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from constants import *

def plotConfusionMatrix(targetsPredList, targetsList, listClasses):
    confusionMatrix = confusion_matrix(targetsPredList, targetsList, labels = list(listClasses.values()))
    confusionMatrixDisplay = ConfusionMatrixDisplay(confusionMatrix, display_labels = list(listClasses.keys()))

    # Set the size of the figure here, which can also help with the overlapping labels. 
    # You may adjust the figure size to suit your needs.
    #fig, ax = plt.subplots(figsize=(10, 10)) 
    confusionMatrixDisplay = confusionMatrixDisplay.plot()
    # Rotate labels on the x-axis for better visibility. Adjust the rotation angle to suit your needs.
    plt.xticks(rotation=45)
    # # This can shift the position of the bottom labels to avoid overlapping. Adjust as needed.
    # plt.subplots_adjust(bottom=0.15)

    plt.show()