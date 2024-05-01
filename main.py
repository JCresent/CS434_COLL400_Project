import pandas as pd
from sklearn.utils import resample

import os

from KNN import *
from NN import *


RAND_ST = 42 # to produce replicable results
KNN_TRAIN_SIZE = 7000


def reformatInfoAndPacket(dataframe):
    protocolMap = {}

    protocols = dataframe["Protocol"].value_counts().index
    infos = dataframe["Info"].value_counts().index
    
    for protocol in protocols:
        if protocol not in protocolMap.keys():
            protocolMap[protocol] = len(protocolMap)

    dataframe["Protocol"] = dataframe["Protocol"].map(protocolMap)
    if "Website" in dataframe.columns:
        return dataframe[["Time", "Protocol", "Length","Website"]]
    else:
        return dataframe[["Time", "Protocol", "Length"]]


def csv_dir_to_df(dir_path):
    dfs = []

    for path in os.listdir(dir_path):
        dfs.append(pd.read_csv(dir_path + "/" + path))

    return pd.concat(dfs)


def get_data(dataset, size=0):
    chatgpt_df      = csv_dir_to_df("Data/" + dataset + "/LinkedIn")
    linkedin_df     = csv_dir_to_df("Data/" + dataset + "/ChatGPT")
    blackboard_df   = csv_dir_to_df("Data/" + dataset + "/Blackboard")
    
    sampleSize = min(len(chatgpt_df), len(blackboard_df), len(linkedin_df))
    
    chatgpt_df      = resample(chatgpt_df, replace=False, n_samples=sampleSize, random_state=RAND_ST)
    linkedin_df     = resample(linkedin_df, replace=False, n_samples=sampleSize, random_state=RAND_ST)
    blackboard_df   = resample(blackboard_df, replace=False, n_samples=sampleSize, random_state=RAND_ST)

    linkedin_df["Website"] = "LinkedIn"
    chatgpt_df["Website"] = "ChatGPT"
    blackboard_df["Website"] = "Blackboard"

    complete_data = pd.concat([chatgpt_df, linkedin_df, blackboard_df])
    numericData = reformatInfoAndPacket(complete_data) 

    if size < 1 or size > len(numericData):
        # invalid size or no size passed in, don't down-sample data
        size = len(numericData)

    return resample(numericData, replace=False, n_samples=size, random_state=RAND_ST)


def main():
    # Use the same test data for all models
    test_data = get_data("Test")

    knn_train_data = get_data("Train", KNN_TRAIN_SIZE)
    run_KNN(knn_train_data, test_data)

    nn_train_data = get_data("Train")
    # run_NN(nn_train_data, test_data)
    

if __name__ == "__main__": 
    main()
