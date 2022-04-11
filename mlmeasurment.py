# Machine Learning measurment functions
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from cellobj import CellObj, add_measured_value, save_objects_as_pickle
import pickle
import numpy as np

def fit_KMeans(data,show_metrics=True):
    '''fitting a clustering algorithm to a traning data'''
    X = np.array(data)
    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    model = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = model.labels_
    return model, scaler

def calculate_label_KMeans(cell_obj, model, scaler, features):
    '''returns a dict of labels from machine learning model'''
    # create feature vector from img_obj:
    values_list = []
    for key, value in cell_obj.measured_values.items():
        if key in features:
            values_list.append(value)
    X = np.array(values_list)
    X = X.reshape(1, -1)
    X = scaler.transform(X)
    label = int(model.predict(X))
    return {'KMeans_label': label}
