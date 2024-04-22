import pandas as pd 

chatGPTcsv = "lakeData/4-8_chatGPT.csv"
blackboardcsv = "lakeData/4-9_blackboard.csv"
linkedIn = "lakeData/4-9_linkedin.csv"

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
