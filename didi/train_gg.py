#! -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv("./train_data/s1_train.csv", sep=",", header=None, names=["district_id",
                                                                                 "time",
                                                                                 "tj_level_1",
                                                                                 "tj_level_2",
                                                                                 "tj_level_3",
                                                                                 "tj_level_4",
                                                                                 "percent"])


def g_time(time_str):
    return int(time_str.split("-")[-1])


def create_submission(predictions, predict_df, loss):
    result1 = pd.DataFrame(predictions, columns=["gap"])

    final_result = pd.concat([predict_df, result1], axis=1)
    #final_result = final_result["district_id", "time", "gap"]
    print "final_result", final_result[:10]
    print "final_result.shape", final_result.shape

    final_result = final_result.loc[:, ["district_id","time", "gap"]]
    final_result = final_result.dropna()
    final_result["district_id"] = final_result["district_id"].astype(int)
    #final_result["district_id"] = int(final_result["district_id"])
    now = datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')

    suffix = str(round(loss, 4)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    final_result.to_csv(sub_file, index=False)


train_df["time"] = [g_time(item) for item in train_df["time"]]
train_df = train_df.dropna()

train_data = train_df.loc[:, ["district_id", "time", "tj_level_1", "tj_level_2", "tj_level_3", "tj_level_4"]]
train_target_data = train_df["percent"]

rfr = RandomForestRegressor(n_estimators=1000, max_depth=13, n_jobs=2, max_features=6)
#clf = rfr.fit(train_data, train_target_data)

est = GradientBoostingRegressor(loss='lad',n_estimators=1000, max_depth=13, learning_rate=0.01)
clf = est.fit(train_data, train_target_data)

# scores = cross_val_score(clf, train_data, train_target_data, cv=3, n_jobs=3)
# print scores

test_df = pd.read_csv("./train_data/s1_test.csv", sep=",", header=None, names=["district_id",
                                                                               "time",
                                                                               "tj_level_1",
                                                                               "tj_level_2",
                                                                               "tj_level_3",
                                                                               "tj_level_4",
                                                                               "percent"])

test_df["time"] = [g_time(item) for item in test_df["time"]]
test_df = test_df.fillna(method='pad')

test_data = test_df.loc[:, ["district_id", "time", "tj_level_1", "tj_level_2", "tj_level_3", "tj_level_4"]]
test_target_data = test_df["percent"]

test_score = clf.score(test_data, test_target_data)

print "test_score: ", test_score
print "mape_socre: ", np.mean(abs(clf.predict(test_data)-test_target_data)/(test_target_data+1))



predict_df = pd.read_csv("./train_data/s1_predict.csv", sep=",", header=None, names=["district_id",
                                                                                     "time",
                                                                                     "tj_level_1",
                                                                                     "tj_level_2",
                                                                                     "tj_level_3",
                                                                                     "tj_level_4"])
good_df = predict_df.copy()
print "good_df.shape", good_df.shape


predict_df["time"] = [g_time(item) for item in predict_df["time"]]
predict_df = predict_df.fillna(method='pad')
good_df = good_df.fillna(method='pad')

predictions = clf.predict(predict_df)

create_submission(predictions, good_df, test_score)
