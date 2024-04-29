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

