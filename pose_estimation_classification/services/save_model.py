import os
from structure.constants import PATH_TO_MODELS

def saveModel(model, modelName):

    jsonModel = model.to_json()
    with open(PATH_TO_MODELS+f"{modelName}.json", "w") as json_file:
        json_file.write(jsonModel)
    # serialize weights to HDF5
    model.save_weights(PATH_TO_MODELS+f"{modelName}.h5")
    print("Saved model to disk")