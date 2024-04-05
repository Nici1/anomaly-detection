import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import itertools
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from sklearn import metrics

from src.algorithms.kmeans import Kmeans
from src.algorithms.dbscan import DBscan
from src.algorithms.isolation_forest import IsolationForest
from src.algorithms.gan import GAN
from src.algorithms.border_check import BorderCheck
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.estimator_checks import check_estimator
import csv
import os
import multiprocessing as mp
import time


def compute_metrics(y_true, y_predicted):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_predicted)
    precision = sklearn.metrics.precision_score(y_true, y_predicted)
    recall = sklearn.metrics.recall_score(y_true, y_predicted)
    f1 = sklearn.metrics.f1_score(y_true, y_predicted)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_predicted)

    return {"confusion matrix ": confusion_matrix, "precision ": precision, "recall ": recall, "f1 ": f1, "accuracy " : accuracy}


class Estimator(BaseEstimator):

    def __init__(
        self,
        
        train_data="data/train/ads-1.csv",
        alg = "Kmeans",

        # Kmeans 
        n_clusters=1,
        treshold=1,

        #DBscan
        eps = 0.5,
        db_treshold=1,
        min_samples = 2,

        #IsolationForest
        max_samples = 100,
        max_features=1,
        contamination = 0.05,

        # GAN
        N_latent=3,
        K=8,
        len_window=500,

        # BorderCheck
        UL=0.5,
        LL=-0.5

        
       
    ):
        self.train_data = train_data
        self.alg = alg

        # Kmeans
        self.n_clusters = n_clusters
        self.treshold = treshold

        # DBscan
        self.eps = eps
        self.min_samples = min_samples
        self.db_treshold = db_treshold

        # IsolationForest
        self.max_samples = max_samples 
        self.max_features = max_features
        self.contamination = contamination

        # GAN
        self.N_latent = N_latent
        self.K = K
        self.len_window = len_window

        # BorderCheck
        self.UL = UL
        self.LL = LL 


    def fit(self, X, y):

        inner_dict = {
            "train_data": self.train_data,
            # Kmeans
            "n_clusters": self.n_clusters,
            "treshold": self.treshold,
            # DBscan
            "eps": self.eps,
            "min_samples": self.min_samples,
            "db_treshold": self.db_treshold,
            # IsolationForest
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "contamination": self.contamination,
            # GAN
            "N_latent": self.N_latent,
            "K": self.K,
            "len_window": self.len_window,
            # BorderCheck
            "UL": self.UL,
            "LL": self.LL,

        }

        conf = {
            "filtering": "None",
            "input_vector_size": 1,
            "warning_stages": [0.7, 0.9],
            "model_name":"IsolationForest",
            **inner_dict,
            "output": [],
            "output_conf": [{}],
        }

      

        class_ = globals()[self.alg]
        self.detector_ = class_(conf)

        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def predict(self, X):
        
        y_predicted = []
                
    
        # transverse X rows
        
        for idx, row in X.iterrows():
            
          
            message = {
                "timestamp": row["timestamp"],
                "ftr_vector": [row["ftr_vector"]],
            }
          
        
            status_code = self.detector_.message_insert(message)

            if status_code == 2 or status_code ==0:
                y_predicted.append(False)
            if status_code == 1:
                y_predicted.append(False)
            elif status_code == -1:
                y_predicted.append(True)

           
       
      

        
        return y_predicted
    



      
def main():

    algorithms = ["IsolationForest"]

    DBscan_params = {
        
        "alg": ["DBscan"],
    
        # DBscan
        "eps": np.arange(0.1, 1, 0.1),
        "db_treshold": np.arange(0.05, 1, 0.05),
        "min_samples": [100, 200],
    
    }

    Kmeans_params = {
        
        "alg": ["Kmeans"],
        # Kmeans
        "n_clusters": [2, 4],
        "treshold": [0.1, 0.15],
    }

    IsolationForest_params = {
        "alg": ["IsolationForest"],
    
        # IsolationForest
        "max_samples": [1500, 3000, 4500],
        "max_features": [1],
        "contamination": np.arange(0.001, 0.01, 0.001),

    }

    BorderCheck_params = {
        "alg": ["BorderCheck"],
    
        "UL": np.arange(0.3, 2.1, 0.2),
        "LL": np.arange(-0.1, -2.1, -0.2),

    }

    first_values = [0.2, 0.5, 0.7, 1, 2]
    second_values = [99, 99.2, 99.5, 99.8, 99.95]  

    # Create a list of tuples with all combinations of values
    list_of_tuples = [(first, second) for first, second in itertools.product(first_values, second_values)]

    Percentile_params = {
        "alg": ["Percentile"],
    
        "percentile_range":list_of_tuples,
        "buff_size": [1000]

    }



    for alg in algorithms:

        for i in range(1, 10):

            df_validation = pd.read_csv(f"data/validation/ads-{i}.csv")
            X_validation = df_validation[["timestamp", "ftr_vector"]]
            y_validation = df_validation[["label"]]


            df_test = pd.read_csv(f"data/test/ads-{i}.csv")
            X_test = df_test[["timestamp", "ftr_vector"]]
            y_test = df_test[["label"]]


            test_params = {
                
                # Kmeans
                "n_clusters": [2],
                "treshold": [0.5],

                # DBscan
                "eps": [0.5],
                "db_treshold": [0.5],
                "min_samples": [100],

                # IsolationForest
                "max_samples": [10],
                "max_features": [1],
                "contamination": [0.01],

                # GAN
                "N_latent": [3],
                "K": [8],
                "len_window": [500],

                # Border Check
                "UL": [0.5],
                "LL":  [-0.5],

                # Percentile
                "percentile_range":[(1,99)],
                "buff_size":[100],

                **eval(f"{alg}_params"),
                "train_data": [f"data/train/ads-{i}.csv"],
        
            }

            print(test_params)

            estimator = Estimator()

            clf = GridSearchCV(
                estimator,
                param_grid=test_params,
                scoring="precision",
                
                cv=TimeSeriesSplit(n_splits=2)
            )

    
            #print(X_validation.shape, y_validation.shape)
            #print(X_validation.index[0], y_validation.index[0])
            selected = clf.fit(X_validation, y_validation)

        
            
            best_estimator = clf.best_estimator_
            y_pred = best_estimator.predict(X_test)
        

            comp_metrics = compute_metrics(y_test, y_pred)
            #print("Metrics ", comp_metrics)

            with open(f"results_1/{alg}_metrics.txt", "a") as file:
            # Write content to the file
                file.write(f"data-{i} {str(best_estimator.get_params())} {str(comp_metrics)}\n")
            
        
            
            transposed_data = zip(*[selected.cv_results_[key] for key in selected.cv_results_])
            is_empty = (
                not os.path.exists(f"results_1/{alg}-precision.csv")
                or os.path.getsize(f"results_1/{alg}-precision.csv") == 0
            )

            with open(f"results_1/{alg}-precision.csv", "a", newline="") as f:
                writer = csv.writer(f)

                # Write headers only if the file is empty
                if is_empty:
                    writer.writerow(selected.cv_results_.keys())

                # Write data rows
                writer.writerows(transposed_data)

    
        





if __name__ == "__main__":
    main()
