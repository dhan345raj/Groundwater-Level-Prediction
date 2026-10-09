import pickle
import numpy as np

model = pickle.load(open("../models/model.pkl", "rb"))

def predict_groundwater(features):
    arr = np.array(features).reshape(1, -1)
    return model.predict(arr)[0]
