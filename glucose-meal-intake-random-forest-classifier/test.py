import numpy as np
import pandas as pd
import pickle
import csv
import os
from datetime import datetime as dt
from datetime import timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

pd.options.mode.chained_assignment = None

# output filepath for Result.csv
pwd = %pwd
pwd = pwd.replace('\\','/')
global out_file
out_file = pwd + '/Result.csv'

################################################################################################################################

def test_model(feats_df):

    filename = "final_model.pkl"
    with open(filename, 'rb') as f:
        model, top_feats = pickle.load(f)

    X = feats_df[top_feats]
    
    global output
    output_array = model.predict(X)
    output = pd.DataFrame(output_array)
    return output.to_csv(out_file, header=None, index=False)
    
################################################################################################################################

def feat_extraction(test_dfs):

    feats_dict = {}
    desc_stats = ['mean', '50%', 'max', 'min', 'std']

    for d in range(len(test_dfs)):

        X = test_dfs[d]

        X['ISIG_diff'] = X['ISIG Value'].diff()
        X['ISIG_diff_2'] = X['ISIG_diff'].diff()
        X['Glucose_diff'] = X['Sensor Glucose (mg/dL)'].diff()
        X['Glucose_diff_2'] = X['Glucose_diff'].diff()

        ISIG = X['ISIG Value']
        up, down, steady = 0, 0, 0
        for x in range(len(ISIG)-1):
            ISIG_now = ISIG.iloc[x]
            ISIG_next = ISIG.iloc[x+1]

            if (ISIG_next > ISIG_now):
                up += 1

            elif (ISIG_next == ISIG_now):
                steady += 1

            elif (ISIG_next < ISIG_now):
                down += 1

            ud_ratio = up/(up+down+steady)

        X['Hour'] = 0.0
        for z in range(len(X)):
            X['Hour'][z:z+1] = np.int64(X['Datetime'].iloc[z].hour)

        feats = X.describe()
        feats = feats.round(3)
        feat_cols = feats.columns

        for col in feat_cols:
            feats_dict[col] = {}

            for stat in desc_stats:
                feat_name = col + ' ' + stat
                feats_dict[col][feat_name] = feats.loc[stat][col]

        cols_list = []
        item_list = []
        for k, v in feats_dict.items():
            for k1, v1 in v.items():
                item_list.append(v1)
                cols_list.append(k1)

        cols_list.extend(['Range_Glucose', 'Range_ISIG', 'ISIG U/D Ratio'])

        range_ISIG = X['ISIG Value'].max() - X['ISIG Value'].min()
        range_Glucose = X['Sensor Glucose (mg/dL)'].max() - X['Sensor Glucose (mg/dL)'].min()

        new_row = pd.DataFrame(item_list).T.values.tolist()
        new_row = new_row[0]
        new_row.extend([range_Glucose, range_ISIG, ud_ratio])

        if (d == 0):
            feats_df = pd.DataFrame(columns=cols_list)

        feats_df.loc[d] = new_row
        if (d % 200 == 0):
            print(str(round(d/len(test_dfs)*100, 0)) + '% of features extracted')
    
    print('100% of features extracted')
    test_model(feats_df)

################################################################################################################################
  
def extract_data(file):
    global df
    df = pd.read_csv(file, header=None, index_col=False)
    df = df.T
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.drop(columns = 'Index', inplace=True)
    df.dropna(axis=1, how='all', inplace = True)
    df = df[::-1].reset_index(drop=True)

    df.drop(columns = ['Index', 'Event Marker', 'Unnamed: 0'], inplace = True, errors='ignore')
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    global test_dfs
    test_dfs = []
    x = 0
    y = 0
    while (x < len(df)):
        y = x+24
        test_dfs.append(df[x:y])
        x = y

    p = 0
    while (p < len(test_dfs)):
        
        test_dfs[p]['Sensor Glucose (mg/dL)'] = pd.to_numeric(test_dfs[p]['Sensor Glucose (mg/dL)'], errors='coerce')
        test_dfs[p]['Sensor Glucose (mg/dL)'].fillna(test_dfs[p]['Sensor Glucose (mg/dL)'].mean(), inplace=True)
        test_dfs[p]['ISIG Value'] = pd.to_numeric(test_dfs[p]['ISIG Value'], errors='coerce')
        test_dfs[p]['ISIG Value'].fillna(test_dfs[p]['ISIG Value'].mean(), inplace=True)
        p += 1
    
    feat_extraction(test_dfs)

################################################################################################################################
    
if __name__ == "__main__":
    
    # input filepaths
    pwd = %pwd
    pwd = pwd.replace('\\','/')
    file = pwd + '/test.csv'

    extract_data(file)
