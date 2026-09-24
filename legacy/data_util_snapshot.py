import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Data"


# pattern: is a regex
def get_files(pattern,path=DATA_PATH):
    files = [f for f in os.listdir(path) if re.match(pattern, f)]
    return files


# cut_precent: cut off pecentage of beginning and end of the simulations
# FIXME: it would be better if create the sequences before concatenate the data, but it should be ok to keep as it is for now
def read_data(file_list=[],path=DATA_PATH, cut_precent=0.05):
    if len(file_list)==0:
        print("No file to read!!")
        return
    # print("vv")
    df = pd.DataFrame(columns=['distance', 'acceleration','displacement', 'force'])
    for f in file_list:
        f_path = os.path.join(path,f)
        df_tmp = pd.read_excel(f_path)
        cut_idx= int(df_tmp.shape[0]*cut_precent)
        df_tmp.columns = ['distance', 'acceleration','displacement', 'force']
        df_tmp = df_tmp.iloc[cut_idx:df_tmp.shape[0]-cut_idx]
        df = pd.concat([df,df_tmp] )
    df = df.dropna()
    df.reset_index(drop=True, inplace=True) 
    return df


# Convert the series to sequences for RNN
def create_sequences(features, target, sequence_length):
        # Normalizing the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    X, y,idx = [], [],[]
    for i in range(len(features) - sequence_length):
        X.append(features[i:(i + sequence_length)])
        y.append(target[i + sequence_length])
        idx.append(target[i + sequence_length])
    return np.array(X), np.array(y),np.array(idx)
