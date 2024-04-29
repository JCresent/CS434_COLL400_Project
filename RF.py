import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.utils import resample


protocolMap = {}
infoMap = {}


def do_Kfold(model,X,y,k,scaler = None, random_state = 146):
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=k, random_state = random_state, shuffle=True)

    train_scores = []
    test_scores = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]
        if scaler != None:
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)

        model.fit(Xtrain,ytrain)

        train_scores.append(model.score(Xtrain,ytrain))
        test_scores.append(model.score(Xtest,ytest))
        
    return train_scores, test_scores

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

#Repalce with actual data 
#TODO
chatGPTcsv = "lakeData/4-8_chatGPT.csv"
blackboardcsv = "lakeData/4-9_blackboard.csv"
linkedIncsv = "lakeData/4-9_linkedin.csv"

data = convertWireSharkData(chatGPTcsv,blackboardcsv,linkedIncsv,3000 )

X = data.drop(columns="Website").values
y = data["Website"].astype("category").values



##########################
#####Creating the RFC#####
##########################
rfc = RFC(random_state = 201) #set state for testing purposes 

#Optimizing value for n_estimators, max_depth, min_sample_splits 
#Using gridsearch
estimators = [5,100,500]
depths = [3,5,9,11,13] #np.arange(2,21)
samples = [3,5,13,20,30,40,70] #np.arange(2,21)

#Create set a dict for params to test 
param_grid = dict(n_estimators=estimators,max_depth = depths,
                 min_samples_split = samples)

#Create object for cross validation 
cv = KFold(n_splits=3, random_state=201, shuffle = True)

#Create the gridsearch 
grid = GridSearchCV(rfc, param_grid=param_grid, cv=cv, n_jobs= -1, scoring='accuracy') #n_jobs = -1 means use all processors, helps speed up GridSearch

#Create a train test split, then fit to grid
Xtrain,Xtest,ytrain,ytest = tts(X,y, test_size= 0.4, random_state = 146)
grid.fit(Xtrain,ytrain)

#See results 
results = pd.DataFrame(grid.cv_results_)[['param_n_estimators','param_max_depth',
                                'param_min_samples_split','mean_test_score','rank_test_score']]
results.head()

print("DONE")
