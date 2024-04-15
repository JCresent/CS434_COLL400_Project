import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


chatGPTcsv = "lakeData/4-8_chatGPT.csv"
blackboardcsv = "lakeData/4-9_blackboard.csv"
linkedIn = "lakeData/4-9_linkedin.csv"


# Format our data
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
numericData = completeData[["Time", "Protocol", "Length", "Website"]]
print(numericData)


# Begin neural network
data = numericData[["Time", "Protocol", "Length"]]
labels = numericData["Website"]

split = train_test_split(data, labels, test_size=0.2)
train_data, test_data, train_labels, test_labels = split

clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(6,), 
                    random_state=1)
clf.fit(train_data, train_labels) 

predictions_train = clf.predict(train_data)
predictions_test = clf.predict(test_data)
train_score = accuracy_score(predictions_train, train_labels)
print("score on train data: ", train_score)
test_score = accuracy_score(predictions_test, test_labels)
print("score on test data: ", test_score)