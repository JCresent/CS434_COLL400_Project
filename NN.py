import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from seaborn import heatmap

from utils import RAND_ST, compare_classes


def pre_process(train_data, test_data):

    X = train_data[["Time", "Protocol", "Length"]]
    y = train_data["Website"]

    X_test = test_data[["Time", "Protocol", "Length"]]
    y_test = test_data["Website"]

    # Scale the data
    scale = StandardScaler()

    # scales, and makes 2d arrays (3 cols)
    X = scale.fit_transform(X)
    X_test = scale.transform(X_test)

    # make y into 1d arrays (1 col)
    y = y.to_numpy()
    y_test = y_test.to_numpy()

    return X, y, X_test, y_test 


def grid_search(X, y):
    mlp = MLPClassifier(random_state=RAND_ST, max_iter=1000)

    # real params for initial large gridsearch
    parameters = {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['adam','lbfgs','sgd'],
        'alpha': 10.0 ** -np.arange(-1, 10),
        'hidden_layer_sizes':[(12),(13),(14),(15),(16),(17),(18),(19),(20),]
        # only testing single layer
        # smallest must be 3 or error, since there are 3 output nodes
    }


    # test params for debug
    # parameters = {
    #     'activation': ['relu'],
    #     'solver': ['adam','lbfgs'],
    #     'alpha': [0.0001],
    #     'hidden_layer_sizes':(9,),
    # }

    gs = GridSearchCV(mlp, parameters, n_jobs=-1, scoring="accuracy", verbose=3)
    gs.fit(X, y)

    # Store and print data for analysis
    print(gs.score(X, y))
    print(gs.best_params_)
    results = pd.DataFrame(gs.cv_results_)
    results = results[['param_activation','param_solver',
                       'param_alpha','param_hidden_layer_sizes',
                       'mean_test_score','std_test_score','rank_test_score']]
    results.to_csv("Output/nn_gridsearch.csv")
    best_params = results[results['rank_test_score'] == 1]
    params_dict = best_params.to_dict(orient='records')[0]
    print(params_dict)

    return gs


def predict_and_show(gs, X_test, y_test):
    test_pred = gs.predict(X_test)
    test_score = accuracy_score(test_pred, y_test)
    print("score on test data: ",test_score)

    y_names = ["ChatGPT","Blackboard","LinkedIn"]
    confusion_matrix, accuracy = compare_classes(y_test, test_pred, y_names)
    print(confusion_matrix)

    heatmap(confusion_matrix, cmap='Purples',
            vmin=0, vmax=800,
            annot=True, fmt='.2f',
            xticklabels=y_names, yticklabels=y_names)
    plt.show()


def run_NN(train_data, test_data):
    print("Starting Neural Network")

    X, y, X_test, y_test = pre_process(train_data, test_data)
    gs = grid_search(X, y)
    predict_and_show(gs, X_test, y_test)

