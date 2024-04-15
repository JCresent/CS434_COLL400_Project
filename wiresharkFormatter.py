import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.utils import resample 
from sklearn.model_selection import train_test_split  as tts
from sklearn.preprocessing import scale 
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.decomposition import PCA


chatGPTcsv = "lakeData/4-8_chatGPT.csv"
blackboardcsv = "lakeData/4-9_blackboard.csv"
linkedIn = "lakeData/4-9_linkedin.csv"



#Format our data
import pandas as pd
chatGPT = pd.read_csv(chatGPTcsv)
print(len(chatGPT))
blackboard = pd.read_csv(blackboardcsv)
print(len(blackboard))
linkedIn = pd.read_csv(linkedIn)

chatGPT["Website"] = 1
blackboard["Website"] = 2
linkedIn["Website"] = 3
completeData = pd.concat([chatGPT, blackboard,linkedIn])


protocols = completeData["Protocol"].value_counts().index
protocolMap = {}
for protocol in protocols:
    if protocol not in protocolMap.keys():
        protocolMap[protocol] = len(protocolMap)
print(protocolMap)
completeData["Protocol"] = completeData["Protocol"].map(protocolMap)
numericData = completeData[["Time", "Protocol", "Length","Website"]]
print(numericData)

