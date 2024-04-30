import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils import resample

protocolMap = {}
# infoMap = {}


def qualifier(model, X, y):
    internalValidation = []
    externalValidation = []
    # KFold object with 10 folds (adjust n_splits as needed)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
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

def reformatInfoAndPacket(dataframe):
    protocols = dataframe["Protocol"].value_counts().index
    infos = dataframe["Info"].value_counts().index
    
    for protocol in protocols:
        if protocol not in protocolMap.keys():
            protocolMap[protocol] = len(protocolMap)
    # for info in infos:
    #     if info not in infoMap.keys():
    #         infoMap[info] = len(infoMap)
    dataframe["Protocol"] = dataframe["Protocol"].map(protocolMap)
    if "Website" in dataframe.columns:
        return dataframe[["Time", "Protocol", "Length","Website"]]
    else:
        return dataframe[["Time", "Protocol", "Length"]]



def convertWireSharkData(chatGPTcsv,blackboardcsv, linkedIncsv, sizeOfDataFrame = 0):
    chatGPT = pd.read_csv(chatGPTcsv)
    #print(len(chatGPT))
    blackboard = pd.read_csv(blackboardcsv)
    #print(len(blackboard))
    linkedIn = pd.read_csv(linkedIncsv)
    sampleSize = min(len(chatGPT),len(blackboard),len(linkedIn))

    blackboard = resample(blackboard,replace=False, n_samples=sampleSize,random_state=42)
    linkedIn = resample(linkedIn,replace=False, n_samples=sampleSize,random_state=42)
    chatGPT = resample(chatGPT,replace=False, n_samples=sampleSize,random_state=42)


    chatGPT["Website"] = "ChatGPT"
    blackboard["Website"] = "Blackboard"
    linkedIn["Website"] = "Linkedin"
    completeData = pd.concat([chatGPT, blackboard,linkedIn])
    numericData = reformatInfoAndPacket(completeData) 
    
    if sizeOfDataFrame == 0:
        sizeOfDataFrame = len(numericData)
    downSampledData = resample(numericData,replace=False, n_samples=sizeOfDataFrame,random_state=42)
    #print(downSampledData)
    return downSampledData



def trainData():
    data = convertWireSharkData("lakeData/train/chatdata.csv","lakeData/train/blackboardData.csv","lakeData/train/linkedindata.csv",7000 )
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








# for index, row in probabilities.iterrows():
#     # Find the index of the column with the maximum probability
#     max_prob_col_index = row.idxmax()

#     # Check if the maximum probability is greater than or equal to the threshold (0.85)
#     if row.max() >= 0.85:
#         # Assign the name of the column with the maximum probability as the prediction
#         prediction = max_prob_col_index
#     else:
#         # If the threshold is not met, assign "Other" as the prediction
#         prediction = "Other"

#     predictions.append(prediction)  # Append prediction to the list


# # Create a new DataFrame with the "Prediction" column
# probabilities_with_prediction = pd.concat([probabilities, pd.Series(predictions, name="Prediction")], axis=1)



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

def testWMData(testDataChatgpt, testDataBlackboard, testDataLinkedIn,model):
    testData = convertWireSharkData(testDataChatgpt,testDataBlackboard,testDataLinkedIn)
    actualClasses = testData["Website"]
    testData = testData.drop("Website", axis = 1)
    predictedClasses = model.predict(testData)
    confusionMatrix, accuracy = compare_classes(actualClasses,predictedClasses)
    cMatrix = confusion_matrix(actualClasses,predictedClasses)
    print(f"Test data contains {len(testData)} packets")


    return confusionMatrix,accuracy,cMatrix


    
    

# we can check what the actual predictions were and see how accurate our model is 
# print(testWMdata)



model = trainData()
plot,modelAccuracy,cmatrix = testWMData("lakeData/test/chatgptTestData.csv", "lakeData/test/blackboardTestData.csv", "lakeData/test/linkedinTestData.csv",model)
print(plot)



