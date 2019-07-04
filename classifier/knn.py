import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  


TRAIND_DATA_PATH = '../data/embeddings.csv'


class KNN:

    def __init__(self, train_data=TRAIND_DATA_PATH, sep=";", n_neighbors=5):
        self.__load_training_data(train_data, sep)
        self.classifier = KNeighborsClassifier(n_neighbors)

    def train_model(self):          
        self.classifier.fit(self.X_train, self.y_train)  

    def __load_training_data(self, train_data, sep):
        names = ["label"] + ["x"+str(n) for n in range(250)]
        dataset = pd.read_csv(train_data, names=names, sep=sep)
        X = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, 0].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20)
        self.__transform_data()

    def __transform_data(self):
        scaler = StandardScaler()  
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)  
        self.X_test = scaler.transform(self.X_test)  

    def get_confusion_matrix(self):
        y_pred = self.classifier.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)
    
    def get_classification_report(self):
        y_pred = self.classifier.predict(self.X_test)
        return classification_report(self.y_test, y_pred)

# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/