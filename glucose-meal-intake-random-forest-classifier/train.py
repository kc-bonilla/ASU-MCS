import numpy as np
import pandas as pd
import sklearn
import pickle
import csv
from datetime import datetime as dt
from datetime import timedelta
import inspect
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

# input filepaths
file = 'C:/Users/KC/Desktop/Project2/CGMData.csv'
file2 = 'C:/Users/KC/Desktop/Project2/InsulinData.csv'
file3 = 'C:/Users/KC/Desktop/Project2/CGM_patient2.csv'
file4 = 'C:/Users/KC/Desktop/Project2/Insulin_patient2.csv'

# output filepath for Result.csv
out_file = 'C:/Users/KC/Desktop/Project2/Result.csv'

meal_dfs = []
no_meal_dfs = []

################################################################################################################################

def train_rfc(feats_df):

    X = feats_df.drop('Target', axis = 1)
    Y = feats_df['Target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state = 0, stratify = Y)
    X_cols = X.columns.tolist()

    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)
    Y_train = np.array(Y_train)
    
    rfc = RandomForestClassifier()

    n_estimators = [100, 400,600,700]
    max_features = ['sqrt']
    max_depth = [10, 15, 20]
    min_samples_split = [2,7,18, 23]
    min_samples_leaf = [2, 7, 13]
    bootstrap = [False]
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    gs = GridSearchCV(rfc, param_grid, cv = 5, verbose = 1, n_jobs=-1)
    gs.fit(X_train_scaled, Y_train)
    rfc = gs.best_estimator_
    gs.best_params_
    gs_df = pd.DataFrame(gs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)

    Y_pred_gs = rfc.predict(X_test_scaled)

    conf_matrix = pd.DataFrame(confusion_matrix(Y_test, Y_pred_gs), index = ['actual 0', 'actual 1'], columns = ['predicted 0', 'predicted 1'])
    display(conf_matrix)
    display('Random Forest Recall Score', recall_score(Y_test, Y_pred_gs))
    
    # Save Model Using Pickle
    filename = 'finalized_model.sav'
    pickle.dump(rfc, open(filename, 'wb'))

################################################################################################################################

    
def feat_extraction(meal_dfs, no_meal_dfs):

    feats_dict = {}
    set_switch = 0
    r = 0
    desc_stats = ['mean', '50%', 'max', 'min', 'std']
    datasets = [no_meal_dfs, meal_dfs]
    
    while (set_switch < 2):
        for set in datasets:
            for y in range(len(set)):
                X = set[y]
                ISIG = X['ISIG Value']

                X['ISIG_diff'] = X['ISIG Value'].diff()
                X['ISIG_diff_2'] = X['ISIG_diff'].diff()
                X['Glucose_diff'] = X['Sensor Glucose (mg/dL)'].diff()
                X['Glucose_diff_2'] = X['Glucose_diff'].diff()

                for z in range(len(X)):
                    X['Hour'] = X['Datetime'].iloc[z].hour

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
                feats = X.describe()
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

                cols_list.extend(['Range_Glucose', 'Range_ISIG', 'ISIG U/D Ratio', 'Target'])

                range_ISIG = X['ISIG Value'].max() - X['ISIG Value'].min()
                range_Glucose = X['Sensor Glucose (mg/dL)'].max() - X['Sensor Glucose (mg/dL)'].min()
                target = set_switch

                new_row = pd.DataFrame(item_list).T.values.tolist()
                new_row = new_row[0]
                new_row.extend([range_Glucose, range_ISIG, ud_ratio, target])

                if (r == 0):
                    feats_df = pd.DataFrame(columns=cols_list)

                feats_df.loc[r] = new_row

                r += 1
            set_switch += 1
    
    train_rfc(feats_df)

################################################################################################################################
  
def extract_data(file, file2, file_switch):

    df = pd.read_csv(file)
    df2 = pd.read_csv(file2)

    df.dropna(axis=1, how='all', inplace = True)
    df2.dropna(axis=1, how='all', inplace = True)

    df = df[::-1].reset_index(drop=True)
    df2 = df2[::-1].reset_index(drop=True)

    df.drop(columns = ['Index', 'Event Marker', 'Unnamed: 0'], inplace = True, errors='ignore')
    df2.drop(columns = ['Index', 'Unnamed: 0'], inplace = True, errors='ignore')

    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df2['Datetime'] = pd.to_datetime(df2['Date'] + ' ' + df2['Time'])

    meal_idxs = df2[(df2['BWZ Carb Input (grams)'] > 0)].index.tolist()
    meal_times = []

    for idx in meal_idxs:

        meal_time = df2.iloc[idx]['Datetime']
        meal_times.append(meal_time)


    good_tms = []
    td = timedelta(hours=2)
    td_thresh = 0.99

    last_time = df2.iloc[-1]['Datetime']

    for j in range(len(meal_times)-1):

        curr_meal_tm = meal_times[j]
        next_meal_tm = meal_times[j+1]

        meal_diff = (next_meal_tm - curr_meal_tm)

        if (meal_diff > (td * td_thresh)):
            good_tms.append(curr_meal_tm)

        if ((next_meal_tm == meal_times[-1]) & ((last_time - next_meal_tm) > (td * td_thresh))):
            good_tms.append(next_meal_tm)

    good_tms.sort()


    k = 0
    meal_tms_dict = {}

    first_cgm_tm = df['Datetime'].min()

    while (k < len(good_tms)):

        ins_meal_tm = good_tms[k]

        if (((ins_meal_tm) - timedelta(minutes = 30)) < first_cgm_tm):
            good_tms.pop(k)
            k = 0

        else:
            meal_tms_dict[k] = {}

            meal_tms_dict[k]['start'] = ins_meal_tm - timedelta(minutes = 30)
            meal_tms_dict[k]['meal'] = ins_meal_tm
            meal_tms_dict[k]['end'] = ins_meal_tm + timedelta(hours = 2)

            k += 1


    meal_data = {}

    for m in range(len(meal_tms_dict)):

        meal_data[m] = df[(df['Datetime'] >= meal_tms_dict[m]['start']) & (df['Datetime'] <= meal_tms_dict[m]['end'])]
        meal_dfs.append(df[(df['Datetime'] >= meal_tms_dict[m]['start']) & (df['Datetime'] <= meal_tms_dict[m]['end'])])

    no_meal_start = meal_tms_dict[0]['end']
    no_meal_end = meal_tms_dict[0]['end']
    last_pt = df['Datetime'].iloc[-1]

    n = 0
    t = 0
    for n in range(len(meal_tms_dict)-1):
        meal_end = meal_tms_dict[n]['end']
        no_meal_start = meal_end
        no_meal_end = meal_end + timedelta(hours=2)

        if (n == len(meal_tms_dict)-1):
            next_meal = last_pt
        else:
            next_meal = meal_tms_dict[n+1]['meal'] 

        while (no_meal_end < next_meal):
            no_meal_dfs.append(df[(df['Datetime'] > no_meal_start) & (df['Datetime'] <= no_meal_end)])

            no_meal_start = no_meal_end
            no_meal_end = no_meal_end + timedelta(hours=2)

            t += 1

        n += 1


    p, q = 0, 0
    while (p < len(meal_dfs)):

        if (meal_dfs[p]['Sensor Glucose (mg/dL)'].notnull().sum() < 20):
            meal_dfs.pop(p)

        elif (meal_dfs[p]['ISIG Value'].notnull().sum() < 20):
            meal_dfs.pop(p)

        else:
            meal_dfs[p]['Sensor Glucose (mg/dL)'].fillna(meal_dfs[p]['Sensor Glucose (mg/dL)'].mean(), inplace=True)
            meal_dfs[p]['ISIG Value'].fillna(meal_dfs[p]['ISIG Value'].mean(), inplace=True)
            p += 1


    while (q < len(no_meal_dfs)):
        if (no_meal_dfs[q]['Sensor Glucose (mg/dL)'].notnull().sum() < 20):
            no_meal_dfs.pop(q)
        elif (no_meal_dfs[q]['ISIG Value'].notnull().sum() < 20):
            no_meal_dfs.pop(q)

        else:
            no_meal_dfs[p]['Sensor Glucose (mg/dL)'].fillna(no_meal_dfs[p]['Sensor Glucose (mg/dL)'].mean(), inplace=True)
            no_meal_dfs[p]['ISIG Value'].fillna(no_meal_dfs[p]['ISIG Value'].mean(), inplace=True)
            q += 1

    if (file_switch == 1):
        feat_extraction(meal_dfs, no_meal_dfs)

################################################################################################################################

def run_files(file, file2, file3, file4):
    
    file_switch = 0
    extract_data(file, file2, file_switch)
    file_switch += 1
    extract_data(file3, file4, file_switch)
    
if __name__ == "__main__":
    run_files(file, file2, file3, file4)
