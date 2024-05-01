import pandas as pd
from sklearn.utils import resample

import os

from KNN import *
from NN import *


RAND_ST = 42 # to produce replicable results
KNN_TRAIN_SIZE = 7000


def clean_data(df, size):
    # make packet column numeric
    protocol_map = {}
    protocols = df["Protocol"].value_counts().index
    
    for protocol in protocols:
        if protocol not in protocol_map.keys():
            protocol_map[protocol] = len(protocol_map)

    df["Protocol"] = df["Protocol"].map(protocol_map)

    # subset to desired rows
    df = df[["Time", "Protocol", "Length", "Website"]]

    if size < 1 or size > len(df):
        # invalid size or no size passed in, don't down-sample data
        size = len(df)

    # potentially down-sample, and mix it up
    return resample(df, replace=False, n_samples=size, random_state=RAND_ST)


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

    combined = pd.concat([chatgpt_df, linkedin_df, blackboard_df])

    return clean_data(combined, size)


def main():
    # Use the same test data for all models
    test_data = get_data("Test")

    knn_train_data = get_data("Train", KNN_TRAIN_SIZE)
    run_KNN(knn_train_data, test_data)

    nn_train_data = get_data("Train")
    # run_NN(nn_train_data, test_data)
    

if __name__ == "__main__": 
    main()
