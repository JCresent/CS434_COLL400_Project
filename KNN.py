import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample

protocolMap = {}
infoMap = {}


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
    for info in infos:
        if info not in infoMap.keys():
            infoMap[info] = len(infoMap)
    dataframe["Protocol"] = dataframe["Protocol"].map(protocolMap)
    dataframe["Info"] = dataframe["Info"].map(infoMap)
    #print(f"protocolMap: {protocolMap}, InfoMap {infoMap}")
    if "Website" in dataframe.columns:
        return dataframe[["Time", "Protocol", "Length","Info","Website"]]
    else:
        return dataframe[["Time", "Protocol", "Length","Info"]]



def convertWireSharkData(chatGPTcsv,blackboardcsv, linkedIncsv, sizeOfDataFrame):
    chatGPT = pd.read_csv(chatGPTcsv)
    #print(len(chatGPT))
    blackboard = pd.read_csv(blackboardcsv)
    #print(len(blackboard))
    linkedIn = pd.read_csv(linkedIncsv)
    chatGPT["Website"] = "ChatGPT"
    blackboard["Website"] = "Blackboard"
    linkedIn["Website"] = "Linkedin"
    completeData = pd.concat([chatGPT, blackboard,linkedIn])
    numericData = reformatInfoAndPacket(completeData) 
    downSampledData = resample(numericData,replace=False, n_samples=sizeOfDataFrame,random_state=42)
    #print(downSampledData)
    return downSampledData

def convertWMdata(wmCSV):
    wmData = pd.read_csv(wmCSV)
    #print(wmData.head())
    return reformatInfoAndPacket(wmData)


chatGPTcsv = "lakeData/4-8_chatGPT.csv"
blackboardcsv = "lakeData/4-9_blackboard.csv"
linkedIncsv = "lakeData/4-9_linkedin.csv"

data = convertWireSharkData(chatGPTcsv,blackboardcsv,linkedIncsv,3000 )


X = data.drop(columns="Website").values
y = data["Website"].astype("category").values

K_range = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37]
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


WMdata = convertWMdata("lakeData/wmwifiData.csv")
class_labels = model.classes_



probabilities = pd.DataFrame(model.predict_proba(WMdata),columns=[class_labels[0],class_labels[1],class_labels[2]])
predictions = [] 


for index, row in probabilities.iterrows():
    # Find the index of the column with the maximum probability
    max_prob_col_index = row.idxmax()

    # Check if the maximum probability is greater than or equal to the threshold (0.85)
    if row.max() >= 0.85:
        # Assign the name of the column with the maximum probability as the prediction
        prediction = max_prob_col_index
    else:
        # If the threshold is not met, assign "Other" as the prediction
        prediction = "Other"

    predictions.append(prediction)  # Append prediction to the list


# Create a new DataFrame with the "Prediction" column
probabilities_with_prediction = pd.concat([probabilities, pd.Series(predictions, name="Prediction")], axis=1)

# Print the modified DataFrame
print(probabilities_with_prediction)


