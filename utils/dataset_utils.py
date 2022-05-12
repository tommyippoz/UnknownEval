import numpy as np
import pandas as pd


def load_tabular_dataset(dataset_name, label_name, limit=np.nan):
    """
    Method to process an input dataset as CSV
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    df = pd.read_csv(dataset_name, sep=",")

    # Shuffle
    df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)

    # Testing Purposes
    if (np.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    # Label encoding with integers
    encoding = pd.factorize(df[label_name])
    y_enc = encoding[0]
    labels = encoding[1]

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("Dataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(len(normal_frame.index)) +
          " normal and " + str(len(labels)) + " labels")
    att_rate = 1-len(normal_frame.index)/len(df.index)

    # Train/Test Split of Classifiers
    y = df[label_name]
    df = df.select_dtypes(exclude=['object'])
    feature_list = df.columns
    df[label_name] = y

    return df, labels, feature_list, att_rate
