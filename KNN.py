import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler #, MinMaxScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold #, LeaveOneOut
from sklearn.metrics import confusion_matrix #, classification_report, ConfusionMatrixDisplay
# from sklearn.utils import resample


RAND_ST = 42 # to produce replicable results


def qualifier(model, X, y):
    internalValidation = []
    externalValidation = []
    # KFold object with 10 folds (adjust n_splits as needed)
    kf = KFold(n_splits=10, shuffle=True, random_state=RAND_ST)
    for train_index, test_index in kf.split(X):
        xtrain, xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        scale = StandardScaler()
        xtrain = scale.fit_transform(xtrain)
        xtest = scale.transform(xtest)
        model.fit(xtrain, ytrain)
        internalValidation.append(model.score(xtrain, ytrain))
        externalValidation.append(model.score(xtest, ytest))
    return np.mean(internalValidation), np.mean(externalValidation)


def trainData(data):
    print(f"Train data contains {len(data)} packets")
    X = data.drop(columns="Website").values
    y = data["Website"].astype("category").values
    K_range = np.linspace(10,30).astype('int64')
    K_range
    internalValiation = []
    externalValidation = []

    for k in K_range:

        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        # we need internalTemp and externalTemp values because you dont want to call the qualifier twice which would waste more time and memory
        internalTemp, externalTemp = qualifier(model,X,y)
        internalValiation.append(internalTemp)
        externalValidation.append(externalTemp)
    print(f" the K range with the highest accuracy {K_range[np.argmax(externalValidation)]}, with an accuracy of: {max(externalValidation)}")
    highestAccuracyK = K_range[np.argmax(externalValidation)]
    model = KNeighborsClassifier(n_neighbors=highestAccuracyK, weights="distance")
    model.fit(X, y)
    print(model.score(X,y))
    return model


def compare_classes(actual, predicted, names=None):
    '''Function returns a confusion matrix, and overall accuracy given:
            Input:  actual - a list of actual classifications
                    predicted - a list of predicted classifications
                    names (optional) - a list of class names
    '''
    accuracy = sum(actual==predicted)/actual.shape[0]
    
    classes = pd.DataFrame(columns = ['Actual', 'Predicted'])
    classes['Actual'] = actual
    classes['Predicted'] = predicted

    conf_mat = pd.crosstab(classes['Actual'], classes['Predicted'])
    
    if type(names) != type(None):
        conf_mat.index = names
        conf_mat.index.name = 'Actual'
        conf_mat.columns = names
        conf_mat.columns.name = 'Predicted'
    
    print('Accuracy = ' + format(accuracy, '.2f'))
    return conf_mat, accuracy


def testWMData(testData, model):
    actualClasses = testData["Website"]
    testData = testData.drop("Website", axis = 1)
    predictedClasses = model.predict(testData)
    confusionMatrix, accuracy = compare_classes(actualClasses,predictedClasses)
    cMatrix = confusion_matrix(actualClasses,predictedClasses)
    print(f"Test data contains {len(testData)} packets")

    return confusionMatrix,accuracy,cMatrix


def run_KNN(train_data, test_data):
    model = trainData(train_data)
    plot,modelAccuracy,cmatrix = testWMData(test_data ,model)
    print(plot)
