import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.utils import resample


protocolMap = {}


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

def make_grid(x_range,y_range):
    '''Function will take a list of x values and a list of y values, 
    and return a list of points in a grid defined by the two ranges'''
    import numpy as np
    xx,yy = np.meshgrid(x_range,y_range)
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    return points

def plot_groups(points, groups, colors, 
               ec='black', ax='None',s=30, alpha=0.5,
               figsize=(6,6)):
    '''Creates a scatter plot, given:
            Input:  points (array)
                    groups (an integer label for each point)
                    colors (one rgb tuple for each group)
                    ec (edgecolor for markers, default is black)
                    ax (optional handle to an existing axes object to add the new plot on top of)
            Output: handles to the figure (fig) and axes (ax) objects
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a new plot, unless something was passed for 'ax'
    if ax == 'None':
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
    
    for i,lbl in enumerate(np.unique(groups)):
        idx = (groups==lbl)
        ax.scatter(points[idx,0], points[idx,1],color=colors[i],
                    ec=ec,alpha=alpha,label = lbl,s=s)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(bbox_to_anchor=[1, 0.5], loc='center left')
    return fig, ax

def reformatInfoAndPacket(dataframe):
    protocols = dataframe["Protocol"].value_counts().index
    
    for protocol in protocols:
        if protocol not in protocolMap.keys():
            protocolMap[protocol] = len(protocolMap)
    dataframe["Protocol"] = dataframe["Protocol"].map(protocolMap)
    if "Website" in dataframe.columns:
        return dataframe[["Time", "Protocol", "Length", "Website"]]
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
    #print(wmData.head())
    return reformatInfoAndPacket(wmData)

#Repalce with actual data 
#TODO
chatGPTcsv = "lakeData/train/4-8_chatGPT.csv"
blackboardcsv = "lakeData/train/4-9_blackboard.csv"
linkedIncsv = "lakeData/train/4-9_linkedin.csv"

data = convertWireSharkData(chatGPTcsv,blackboardcsv,linkedIncsv)

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

print(results[results['rank_test_score'] == 1])

rfc_final = RFC(n_estimators=100, max_depth=9, min_samples_split=5, random_state=201)
rfc_final.fit(Xtrain, ytrain)
print('Train Score: ', rfc_final.score(Xtrain, ytrain))
print('Test Score: ', rfc_final.score(Xtest, ytest))

#Confusion matrix 
y_names = ["ChatGPT","Blackboard","LinkedIn"]
cnf_matrix, accuracy = compare_classes(ytest, rfc_final.predict(Xtest), y_names)
print(cnf_matrix, accuracy)

#Make heatmap visual 
sns.heatmap(cnf_matrix, cmap='Purples', vmin=0, vmax=800,
            annot=True, fmt='.2f', xticklabels=y_names, yticklabels=y_names)
plt.show()

#Test on validation sets
test_data = convertWireSharkData("lakeData/test/chatgptTestData.csv", "lakeData/test/blackboardTestData.csv", "lakeData/test/linkedinTestData.csv")
X_test = test_data.drop(columns="Website").values
y_test = test_data["Website"]
cnf_matrix_tst, accuracy_tst = compare_classes(y_test, rfc_final.predict(X_test), y_names)
print(cnf_matrix_tst, accuracy_tst)

#Make heatmap visual 
sns.heatmap(cnf_matrix_tst, cmap='Purples', vmin=0, vmax=800,
            annot=True, fmt='.2f', xticklabels=y_names, yticklabels=y_names)
plt.show()
#print("DONE")
