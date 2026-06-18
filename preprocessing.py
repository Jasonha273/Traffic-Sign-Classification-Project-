import numpy as np
from sklearn.utils import shuffle
from skimpy import exposure

def preprocess_data(X, y=None):
    #Convert to grayscale
    weights = [0.299,0.587,0.114]
    np.dot(X, weights)

    #Normalize pixel values to [0,1]
    X = (X / 255.0).astype(np.float32)

    #Apply histogram equalization
    for i in range(X.shape[0]):
        X[i] = exposure.equalize_adapthist(X[i])
    if y is not None:
        #Shuffle data
        y=np.eye(43)[y]
        X, y = shuffle(X, y, random_state=42)
    X= X.reshape(X.shape +(1,))
    return X, y