#! -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import cross_val_score

train_df = pd.read_csv("./train_data/s1_train.csv", sep=",", header=None, names=["district_id",
                                                                                 "time",
                                                                                 "tj_level_1",
                                                                                 "tj_level_2",
                                                                                 "tj_level_3",
                                                                                 "tj_level_4",
                                                                                 "percent"])


def g_time(time_str):
    return int(time_str.split("-")[-1])


train_df["time"] = [g_time(item) for item in train_df["time"]]
train_df = train_df.dropna()

train_data = train_df.loc[:, ["district_id", "time", "tj_level_1", "tj_level_2", "tj_level_3", "tj_level_4"]]
train_target_data = train_df["percent"]

clf = RandomForestRegressor(n_estimators=1000, max_depth=15)
scores = cross_val_score(clf, train_data, train_target_data, cv=8, n_jobs=3,scoring='accuracy')
print scores
test_df = pd.read_csv("./train_data/s1_test.csv", sep=",", header=None, names=["district_id",
                                                                                 "time",
                                                                                 "tj_level_1",
                                                                                 "tj_level_2",
                                                                                 "tj_level_3",
                                                                                 "tj_level_4",
                                                                                 "percent"])
test_df = test_df.dropna()
test_data = train_df.loc[:, ["district_id", "time", "tj_level_1", "tj_level_2", "tj_level_3", "tj_level_4"]]
test_target_data = train_df["percent"]

test_predict_data = clf.predict(test_data)
print np.mean((test_predict_data - test_target_data)/test_target_data)
