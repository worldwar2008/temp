#! -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import fbeta_score, make_scorer

train_df = pd.read_csv("./train_data/s1_train_atw.csv", sep=",", header=None, names=["district_id",
                                                                                     "time",
                                                                                     "percent",
                                                                                     "tj_level_1",
                                                                                     "tj_level_2",
                                                                                     "tj_level_3",
                                                                                     "tj_level_4",
                                                                                     "weather",
                                                                                     "temperature",
                                                                                     "pm25"
                                                                                     ])


def g_time(time_str):
    return int(time_str.split("-")[-1])


def get_weekday(date_str):
    [y, m, d, other] = date_str.split("-")
    date_str = y + "-" + m + "-" + d
    d = datetime.strptime(date_str, "%Y-%m-%d")
    result = d.weekday()
    if (result == 5) | (result == 6):
        return 1
    else:
        return 0


def create_submission(predictions, predict_df, loss):
    result1 = pd.DataFrame(predictions, columns=["gap"])

    final_result = pd.concat([predict_df, result1], axis=1)
    # final_result = final_result["district_id", "time", "gap"]
    print "final_result", final_result[:10]
    print "final_result.shape", final_result.shape

    final_result = final_result.loc[:, ["district_id", "time", "gap"]]
    final_result = final_result.dropna()
    final_result["district_id"] = final_result["district_id"].astype(int)
    # final_result["district_id"] = int(final_result["district_id"])
    now = datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')

    suffix = str(round(loss, 4)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    final_result.to_csv(sub_file, index=False)


train_df["week"] = [get_weekday(item) for item in train_df["time"]]
train_df["time"] = [g_time(item) for item in train_df["time"]]
train_df = train_df.sort_values(by=["district_id", "time"])
train_df = train_df.fillna(method='pad')
train_df = train_df.fillna(method='backfill')
# train_df = train_df.dropna()

train_data = train_df.loc[:, ["district_id",
                              "time",
                              "week",
                              "tj_level_1",
                              "tj_level_2",
                              "tj_level_3",
                              "tj_level_4",
                              "weather",
                              "temperature",
                              "pm25"]]

train_target_data = train_df["percent"]

# rfr = RandomForestRegressor(n_estimators=2000, max_depth=16, n_jobs=4, max_features=8)
rfr = RandomForestRegressor(n_estimators=8000, n_jobs=3, max_features=5)

clf = rfr.fit(train_data, train_target_data)


# est = GradientBoostingRegressor(loss='lad', n_estimators=1000, max_depth=13, learning_rate=0.5)
# clf = est.fit(train_data, train_target_data)

def get_mape(test_df):
    test_predict = clf.predict(test_data)
    # old_mape = np.mean(np.abs(clf.predict(test_data) - test_target_data) / (test_target_data + 1))
    test_df["predict"] = test_predict
    test_df["gap_diff"] = np.abs(test_df["predict"] - test_df["percent"]) / (test_df["percent"] + 1.0)
    test_df = test_df.fillna("pad")
    return np.mean(test_df.groupby(["district_id"])["gap_diff"].mean())


def my_mape_loss_func(ground_truth, predictions):
    diff = np.mean(np.abs(ground_truth - predictions) / np.abs(ground_truth + 1.0))
    #print "guo_diff:", diff
    return np.abs(diff)


def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)


mape_loss = make_scorer(my_mape_loss_func, greater_is_better=False)

# val_scores = cross_val_score(rfr, train_data, train_target_data, cv=3, n_jobs=4, scoring="mean_absolute_error")
val_scores = cross_val_score(rfr, train_data, train_target_data, cv=3, n_jobs=6, scoring=mape_loss)

print "val_scores", val_scores
print "val_scores_mean", np.mean(val_scores)

test_df = pd.read_csv("./train_data/s1_test_atw.csv", sep=",", header=None, names=["district_id",
                                                                                   "time",
                                                                                   "percent",
                                                                                   "tj_level_1",
                                                                                   "tj_level_2",
                                                                                   "tj_level_3",
                                                                                   "tj_level_4",
                                                                                   "weather",
                                                                                   "temperature",
                                                                                   "pm25"])
test_df["week"] = [get_weekday(item) for item in test_df["time"]]
test_df["time"] = [g_time(item) for item in test_df["time"]]
test_df = test_df.sort_values(by=["district_id", "time"])
test_df = test_df.fillna(method='pad')
test_df = test_df.fillna(method='backfill')
# test_df = test_df.dropna()

test_data = test_df.loc[:, ["district_id",
                            "time",
                            "week",
                            "tj_level_1",
                            "tj_level_2",
                            "tj_level_3",
                            "tj_level_4",
                            "weather",
                            "temperature",
                            "pm25"]]

test_target_data = test_df["percent"]

test_score = clf.score(test_data, test_target_data)

# clf = clf.fit(train_data, train_target_data)

print "test_score: ", test_score
print "mape_socre: ", get_mape(test_df)

predict_df = pd.read_csv("./train_data/s1_predict_atw.csv", sep=",", header=None, names=["district_id",
                                                                                         "time",
                                                                                         "tj_level_1",
                                                                                         "tj_level_2",
                                                                                         "tj_level_3",
                                                                                         "tj_level_4",
                                                                                         "weather",
                                                                                         "temperature",
                                                                                         "pm25"])
predict_df["week"] = [get_weekday(item) for item in predict_df["time"]]

good_df = predict_df.copy()
print "good_df.shape", good_df.shape

predict_df["time"] = [g_time(item) for item in predict_df["time"]]
## 按照地区和时间进行一个排序
predict_df = predict_df.sort_values(by=["district_id", "time"])
predict_df = predict_df.fillna(method='pad')
good_df = good_df.fillna(method='pad')

predictions = clf.predict(predict_df)
create_submission(predictions, good_df, test_score)
