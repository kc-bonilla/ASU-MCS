import os
from math import *
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, normalize

pd.options.mode.chained_assignment = None

# LOAD INPUT FILES #
PWD = os.path.abspath(os.getcwd()).replace('\\', '/')
FILE = PWD + '/CGMData.csv'
FILE2 = PWD + '/InsulinData.csv'
OUT_FILE = PWD + '/Result.csv'

DT = 'Datetime'
CARBS = 'BWZ Carb Input (grams)'
GLUCOSE = 'Sensor Glucose (mg/dL)'
DESC_STATS = ['mean', '50%', 'max', 'min', 'std']

BIN_STEP = 20
NON_NULL_MIN = 20
TD_THRESH = 0.99
MEAL_TD = timedelta(hours=2)
MEAL_BUFFER_TD = timedelta(minutes=30)
REQUIRED_BREAK = MEAL_TD * TD_THRESH

PCA_N = 2
PCA_COLS = ['P1', 'P2']

DB_DIST = 0.18
N_NEIGHBORS = 23
SS = StandardScaler()

KM_CONFIG = KMeans(
    n_clusters=6,
    init='random',
    n_init=1,
    max_iter=30000,
    tol=1e-04,
    random_state=0
)


class ScaleNormPCA:

    def __init__(self, df, method, apply_pca=True):
        self.x = df
        self.method = method
        self.pca = PCA(n_components=PCA_N)
        if method == km:
            self.x_scaled = self.method.fit(self.x)
        else:
            self.x_scaled = self.method.fit_transform(self.x)
        self.x_norm = normalize(self.x_scaled)
        self.x_final = self.x_norm
        if apply_pca:
            self.x_pca = self.pca.fit_transform(self.x_final)
            self.x_final = pd.DataFrame(self.x_pca)
            self.x_final.columns = PCA_COLS


class DataFrames:

    invalid_cols = [
        'Index', 'Event Marker',
        'Unnamed: 0', 'ISIG Value',
        'Sensor Exception', 'Date',
        'Time'
    ]

    invalid_feats = [
        'Carbs',
        'bin'
    ]

    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(self.file)
        self.drop_na_cols()
        self.final_df = self.drop_df_cols(
            df=self.df,
            cols=DataFrames.invalid_cols
        )

    def drop_na_cols(self):
        self.df.dropna(axis=1, how='all', inplace=True)
        return self.reset_index()

    def reset_index(self):
        self.df = self.df[::-1].reset_index(drop=True)
        return self.create_dt_col()

    def create_dt_col(self):
        self.df[DT] = pd.to_datetime(
            self.df['Date'] + ' ' + self.df['Time'])

    @staticmethod
    def drop_df_cols(df, cols):
        df.drop(
            columns=cols,
            inplace=True,
            errors='ignore')
        return df


class Constants:

    n_bins = None
    bins = None
    letter_lsts = a, b, c, d, e, f = ([],) * 6
    lsts = [letter_lsts]

    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.bins = self.get_bins()
        self.lsts = self.create_lists()
        Constants.n_bins = self.n_bins

    def get_bins(self):
        bins = [_ for _ in range(self.n_bins)]
        Constants.bins = bins
        return bins

    @classmethod
    def create_lists(cls):
        return cls.lsts


class Totals:
    km_sse_total = None
    km_entropy_total = None
    km_purity_total = None
    db_sse_total = None
    db_entropy_total = None
    db_purity_total = None


# OUTPUT RESULTS #
def output_results():
    result = pd.DataFrame([
        Totals.km_sse_total,
        Totals.db_sse_total,
        Totals.km_entropy_total,
        Totals.db_entropy_total,
        Totals.km_purity_total,
        Totals.db_purity_total], columns=None).T
    return result.to_csv(OUT_FILE, header=False, index=False)


# DBSCAN PURITY #
def dbscan_purity(y_pred, y_true):

    y_voted_labels = np.zeros(y_true.shape)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(
            a=y_true[y_pred == cluster], bins=Constants.n_bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
        Totals.db_purity_total = accuracy_score(y_true, y_voted_labels)
    return output_results()


# DBSCAN ENTROPY #
def dbscan_entropy(feats_df):

    bin_feat = y_true = feats_df['bin']
    x = DataFrames.drop_df_cols(
        df=feats_df, cols=DataFrames.invalid_feats)
    x_principal = ScaleNormPCA(
        df=x, method=SS).x_final
    db = DBSCAN(
        eps=DB_DIST, min_samples=N_NEIGHBORS).fit(x_principal)
    y_pred = feats_df['db_pred'] = db.labels_

    n, db_entropy_total = 0, 0
    cluster_entropy = []
    for _ in range(len(Constants.lsts)):
        ct = list(y_pred).count(n)
        for bn in Constants.bins:
            if ct > 0:
                pct = round(len(feats_df[(bin_feat == n) & (y_pred == bn)]) / ct, 4)
                if pct > 0:
                    val = pct * log(pct)
                    cluster_entropy.append(val)
                    db_entropy_total -= val
            else:
                pass
        n += 1
        Totals.db_entropy_total = db_entropy_total
    return dbscan_purity(y_pred, y_true)


# DBSCAN SSE #
def dbscan_sse(feats_df):

    x = DataFrames.drop_df_cols(
        df=feats_df, cols=DataFrames.invalid_feats)
    x_principal = ScaleNormPCA(
        df=x, method=SS).x_final
    db = DBSCAN(
        eps=DB_DIST, min_samples=N_NEIGHBORS).fit(x_principal)
    x_cluster = x_principal['cluster'] = db.labels_

    db_clusters, db_centroids = {}, {}
    # separate ground truth clusters
    for n in range(Constants.n_bins):
        db_clusters[('cluster_' + str(n))] = x_principal[x_cluster == n]
    # calculate centroids
    for key in db_clusters:
        db_centroids[key] = {
            (db_clusters.get(key).get('P1').mean(),
             db_clusters.get(key).get('P2').mean())
        }
    # iterate through each cluster and calculate cluster SSE to get total SSE
    db_sse_total = 0
    for key, group in zip(db_clusters, db_centroids):
        cluster = db_clusters.get(key)
        cent = db_centroids.get(group)
        sse_cluster, j = 0, 0
        while j < len(cluster):
            a = np.array(
                (cluster.iloc[j]['P1'],
                 cluster.iloc[j]['P2'])
            )
            b = np.array((cent[0], cent[1]))
            sse_cluster += abs(np.linalg.norm(b - a))**2
            j += 1
        db_sse_total += sse_cluster
    Totals.db_sse_total = db_sse_total
    return dbscan_entropy(feats_df)


# K-MEANS ENTROPY, PURITY #
def kmeans_entropy_purity(feats_df, km):

    n_bins = Constants.n_bins
    x = DataFrames.drop_df_cols(
        df=feats_df, cols=DataFrames.invalid_feats)
    x_principal = ScaleNormPCA(
        df=x, method=km, apply_pca=False).x_final
    km_pred = feats_df['km_pred'] = km.predict(x_principal)
    bin_feat = y_true = feats_df['bin']

    n, km_entropy_total = 0, 0
    cluster_entropy = []
    for _ in Constants.lsts:
        ct = list(y_pred).count(n)
        for bn in bins:
            if ct > 0:
                pct = round(len(feats_df[(bin_feat == n) and (km_pred == bn)])/ct, 4)
                if pct > 0:
                    val = pct * log(pct)
                    cluster_entropy.append(val)
                    km_entropy_total -= val
            else:
                pass
        n += 1
    Totals.km_entropy_total = km_entropy_total

    y_voted_labels = np.zeros(y_true.shape)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(
            a=y_true[y_pred == cluster], bins=n_bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
    Totals.km_purity_total = accuracy_score(y_true, y_voted_labels)
    return dbscan_sse(feats_df)


# K-MEANS SSE #
def kmeans_sse(feats_df):

    km = KM_CONFIG
    x = DataFrames.drop_df_cols(
        df=feats_df, cols=DataFrames.invalid_feats)
    x_principal = ScaleNormPCA(
        df=x, method=km, apply_pca=False).x_final
    km.fit(x_principal.x_norm)
    Totals.km_sse_total = km.inertia_
    return kmeans_entropy_purity(feats_df, km)


# BINNING FEATURE DF #
def bin_meals(meal_dfs, feats_df):

    all_meals = pd.concat(meal_dfs)
    carb_max = all_meals['Carbs'].max()
    carb_min = all_meals[all_meals['Carbs'] > 0]['Carbs'].min()
    range_carbs = carb_max - carb_min

    n_bins = Constants.n_bins = int(round(range_carbs / BIN_STEP, 0))
    bins = [int(carb_max)]
    for n in range(n_bins):
        cut = int(carb_min + (BIN_STEP * n))
        bins.append(cut)

    labels = []
    bins.sort()
    for b in range(len(bins)-1):
        labels.append(str(bins[b]) + '-' + str(bins[b + 1]))

    feats_df['bin'] = pd.cut(
        feats_df['Carbs'],
        bins=bins,
        labels=[0, 1, 2, 3, 4, 5],
        include_lowest=True
    )
    return kmeans_sse(feats_df)


# FEATURE EXTRACTION #
def feature_extraction(meal_dfs):

    r = 0
    feats_df = None
    feats_dict = {}
    first_loop = bool(r == 0)
    for y in range(len(meal_dfs)):
        x = DataFrames.drop_df_cols(
            df=meal_dfs[y], cols=['Carbs'])
        carbs_mean = int(meal_dfs[y]['Carbs'].mean())
        x['glucose_diff'] = x[GLUCOSE].diff()
        x['glucose_diff_2'] = x['glucose_diff'].diff()

        up, down, steady = 0, 0, 0
        up_streak, down_streak = 0, 0
        max_up_streak, max_down_streak = 0, 0
        for x in range(len(glucose)-1):
            glucose_now = glucose.iloc[x]
            glucose_next = glucose.iloc[x+1]

            glucose_increase = bool(glucose_next > glucose_now)
            glucose_decrease = bool(glucose_next < glucose_now)
            glucose_same = bool(glucose_next == glucose_now)

            new_max_up_streak = bool(up_streak > max_up_streak)
            new_max_down_streak = bool(down_streak > max_down_streak)

            if glucose_increase:
                down_streak = 0
                up_streak += 1
                up += 1
                if new_max_up_streak:
                    max_up_streak = up_streak
            elif glucose_same:
                steady += 1
            elif glucose_decrease:
                up_streak = 0
                down_streak += 1
                down += 1
                if new_max_down_streak:
                    max_down_streak = down_streak

        ud_ratio = up / (up + down + steady)
        streak_ratio = max_up_streak / (max_up_streak + max_down_streak)

        feats = x.describe()
        feats = feats.round(3)
        feat_cols = feats.columns
        for col in feat_cols:
            feats_dict[col] = {}
            for stat in DESC_STATS:
                feat_name = col + ' ' + stat
                feats_dict[col][feat_name] = feats.loc[stat][col]

        cols_list, item_list = [], []
        for k, v in feats_dict.items():
            for k1, v1 in v.items():
                item_list.append(v1)
                cols_list.append(k1)

        cols_list.extend(
            ['Glucose Range',
             'Glucose Variance',
             'Glucose U/D Ratio',
             'Glucose Up Streak',
             'Glucose Down Streak',
             'Glucose Streak Ratio',
             'Carbs']
        )
        g_var = round(feats[GLUCOSE]['std']**2, 3)
        range_glucose = x[GLUCOSE].max() - x[GLUCOSE].min()
        new_row = pd.DataFrame(item_list).T.values.tolist()
        new_row = new_row[0]
        new_row.extend([
            range_glucose,
            g_var,
            ud_ratio,
            max_up_streak,
            max_down_streak,
            streak_ratio,
            carbs_mean
        ])
        if first_loop:
            feats_df = pd.DataFrame(columns=cols_list)
        feats_df.loc[r] = new_row
        r += 1
    return bin_meals(meal_dfs, feats_df)


# EXTRACT MEAL DATA #
def extract_meal_data(df, df2):

    k = 0
    meal_times_dict, meal_data = {}, {}
    meal_dfs, meal_times, good_times = ([],)*3

    meal_activity = df2[df2[CARBS] > 0]
    meal_idxs = meal_activity.index.tolist()
    for idx in meal_idxs:
        meal_time = df2.iloc[idx][DT]
        meal_times.append(meal_time)

    last_time = df2.iloc[-1][DT]
    potential_meal_times = len(meal_times)
    for j in range(potential_meal_times-1):
        curr_meal_time = meal_times[j]
        next_meal_time = meal_times[j+1]
        final_meal_time = bool(next_meal_time == meal_times[-1])

        meal_diff = next_meal_time - curr_meal_time
        final_diff = last_time - next_meal_time

        if meal_diff > REQUIRED_BREAK:
            good_times.append(curr_meal_time)

        if final_meal_time and final_diff > REQUIRED_BREAK:
            good_times.append(next_meal_time)

    good_times.sort()
    first_cgm_time = df[DT].min()
    while k < len(good_times):
        insulin_meal_time = good_times[k]
        if insulin_meal_time - MEAL_BUFFER_TD < first_cgm_time:
            good_times.pop(k)
            k = 0
        else:
            meal_times_dict[k] = {}
            meal_times_dict[k]['meal'] = insulin_meal_time
            meal_times_dict[k]['start'] = insulin_meal_time - MEAL_BUFFER_TD
            meal_times_dict[k]['end'] = insulin_meal_time + MEAL_TD
            k += 1

    total_meals = len(meal_times_dict)
    total_meal_dfs = len(meal_dfs)
    for m in range(total_meals):
        meal_time = meal_times_dict[m]['meal']
        meal_start = meal_times_dict[m]['start']
        meal_end = meal_times_dict[m]['end']
        time_ = df[DT]

        meal_data[m] = df[(time_ >= meal_start) & (time_ <= meal_end)]
        meal_df = df[(time_ >= meal_start) & (time_ <= meal_end)]
        meal_df['Carbs'] = df2[time_ == meal_time][CARBS].max()
        meal_dfs.append(meal_df)

    p = 0
    while p < total_meal_dfs:
        meal_glucose_attr = meal_dfs[p][GLUCOSE]
        if meal_glucose_attr.notnull().sum() < NON_NULL_MIN:
            meal_dfs.pop(p)
        else:
            meal_glucose_attr.fillna(
                meal_glucose_attr.mean(), inplace=True)
            p += 1
    return feature_extraction(meal_dfs)


# GENERATE DATAFRAMES #
def create_dfs(file, file2):

    df, df2 = DataFrames(file).final_df, DataFrames(file2).final_df
    return extract_meal_data(df, df2)


# INITIALIZE #
if __name__ == "__main__":
    create_dfs(FILE, FILE2)
