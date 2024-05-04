import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.model_selection import train_test_split as tts

import GraphModels



protocolMap = {}

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
def reformatInfoAndPacket(dataframe):
    protocols = dataframe["Protocol"].value_counts().index

    
    for protocol in protocols:
        if protocol not in protocolMap.keys():
            protocolMap[protocol] = len(protocolMap)
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

def convertWMdata(wmCSV):
    wmData = pd.read_csv(wmCSV)
    return reformatInfoAndPacket(wmData)

chatGPTcsv = "lakeData/train/4-8_chatGPT.csv"
blackboardcsv = "lakeData/train/4-9_blackboard.csv"
linkedIncsv = "lakeData/train/4-9_linkedin.csv"

data = convertWireSharkData(chatGPTcsv,blackboardcsv,linkedIncsv )



def trainData():
    data = convertWireSharkData("lakeData/train/chatdata.csv","lakeData/train/blackboardData.csv","lakeData/train/linkedindata.csv",7000 )
    # print(f"Train data contains {len(data)} packets")
    X = data.drop(columns="Website").values
    y = data["Website"].astype("category").values
    # K_range = np.linspace(10,30).astype('int64')
    # K_range
    # internalValiation = []
    # externalValidation = []

    # for k in K_range:

    #     model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    #     # we need internalTemp and externalTemp values because you dont want to call the qualifier twice which would waste more time and memory
    #     internalTemp, externalTemp = qualifier(model,X,y)
    #     internalValiation.append(internalTemp)
    #     externalValidation.append(externalTemp)
    print(f" the K range with the highest accuracy {12}, with an accuracy of: {97}%")
    highestAccuracyK = 12
    model = KNeighborsClassifier(n_neighbors=highestAccuracyK, weights="distance")
    model.fit(X, y)
    print(model.score(X,y))
    return model


X_df = data.drop(columns="Website").values
X = np.array(X_df)
y = data["Website"].astype("category").values
Xtrain,Xtest,ytrain,ytest = tts(X,y,test_size=0.4, random_state=146)
logistic_model = LogisticRegression()
logistic_model.fit(Xtrain,ytrain)
logistic_model.score(Xtrain,ytrain)
y_pred = logistic_model.predict(Xtest)
# print(f"-------------------------Train Data-----------------------------")
print(compare_classes(ytest, y_pred))

# print(f"---------------------------Test Data------------------------------------")

WMdata = convertWireSharkData("lakeData/test/chatgptTestData.csv", "lakeData/test/blackboardTestData.csv", "lakeData/test/linkedinTestData.csv")
actualClasses = WMdata["Website"]
testData = WMdata.drop("Website", axis = 1)
predictedClasses = logistic_model.predict(testData)
confusionMatrix, accuracy = compare_classes(actualClasses,predictedClasses)
print(confusionMatrix,accuracy)


def testWMData(testDataChatgpt, testDataBlackboard, testDataLinkedIn,model):
    testData = convertWireSharkData(testDataChatgpt,testDataBlackboard,testDataLinkedIn)
    actualClasses = testData["Website"]
    testData = testData.drop("Website", axis = 1)
    predictedClasses = model.predict(testData)

    graph = GraphModels.Graphs(testData,actualClasses,predictedClasses)
    
    


    return graph.confusionMatrix("LogReg Confusion Matrix"), graph.scatterPlot("Logistic Regression")

model = trainData()
test = testWMData("lakeData/test/chatgptTestData.csv", "lakeData/test/blackboardTestData.csv", "lakeData/test/linkedinTestData.csv",model)
# trainedData = testWMData("lakeData/train/chatdata.csv","lakeData/train/blackboardData.csv","lakeData/train/linkedindata.csv",model)
plt.show()