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


class TrainDivide(object):
    def __init__(self):
        self.est = GradientBoostingRegressor(n_estimators=2000, max_depth=13, learning_rate=0.1)
        self.rfr = RandomForestRegressor(n_estimators=10, n_jobs=1, min_samples_split=3, max_features=5)
        self.test_score = 0
        self.mp = 0

    @staticmethod
    def g_time(time_str):
        return int(time_str.split("-")[-1])

    @staticmethod
    def get_weekday(date_str):
        [y, m, d, other] = date_str.split("-")
        date_str = y + "-" + m + "-" + d
        d = datetime.strptime(date_str, "%Y-%m-%d")
        result = d.weekday()
        if (result == 5) | (result == 6):
            return 1
        else:
            return 0

    def create_submission(self, predictions, predict_df, loss):
        print "len(predictions)", len(predictions)
        print "len(predict_df)", len(predict_df)
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

    def get_train_data(self, district_id):
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
        train_df["week"] = [self.get_weekday(item) for item in train_df["time"]]
        train_df["time"] = [self.g_time(item) for item in train_df["time"]]
        train_df = train_df.sort_values(by=["district_id", "time"])
        train_df = train_df.fillna(method='pad')
        train_df = train_df.fillna(method='backfill')
        # train_df = train_df.dropna()

        train_df = train_df[train_df.district_id == district_id]
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
        return train_data, train_target_data

    def get_test_data(self, district_id):
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
        test_df["week"] = [self.get_weekday(item) for item in test_df["time"]]
        test_df["time"] = [self.g_time(item) for item in test_df["time"]]
        test_df = test_df.sort_values(by=["district_id", "time"])
        test_df = test_df.fillna(method='pad')
        test_df = test_df.fillna(method='backfill')
        # test_df = test_df.dropna()
        test_df = test_df[test_df.district_id == district_id]
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
        return test_data, test_target_data

    def cross_val(self, train_data, train_target_data):
        mape_loss = make_scorer(self.my_mape_loss_func, greater_is_better=False)
        # val_scores = cross_val_score(rfr, train_data, train_target_data, cv=3, n_jobs=4, scoring="mean_absolute_error")
        val_scores = cross_val_score(self.est, train_data, train_target_data, cv=10, n_jobs=6, scoring=mape_loss)
        print "val_scores", val_scores
        print "val_scores_mean", np.mean(val_scores)

    def train_process(self, train_data, train_target_data, test_data, test_target_data):
        # clf = rfr.fit(train_data, train_target_data)
        clf = self.est.fit(train_data, train_target_data)
        self.test_score = clf.score(test_data, test_target_data)
        print "test_score: ", self.test_score

        test_predict = clf.predict(test_data)
        test_data["predict"] = test_predict
        test_data["gap_diff"] = np.abs(test_data["predict"] - test_target_data) / (test_target_data + 0.1)
        test_df = test_data.fillna("pad")
        mape = np.mean(test_df.groupby(["district_id"])["gap_diff"].mean())
        self.mp = mape
        print "mape_socre: ", mape
        return clf

    def my_mape_loss_func(self, ground_truth, predictions):
        diff = np.mean(np.abs(ground_truth - predictions) / np.abs(ground_truth + 0.5))
        # print "guo_diff:", diff
        return np.abs(diff)

    def my_custom_loss_func(self, ground_truth, predictions):
        diff = np.abs(ground_truth - predictions).max()
        return np.log(1 + diff)

    def get_mape(self, test_data, test_target_data):
        test_predict = self.clf.predict(test_data)
        test_data["predict"] = test_predict
        test_data["gap_diff"] = np.abs(test_data["predict"] - test_target_data) / (test_target_data + 0.5)
        test_df = test_data.fillna("pad")
        return np.mean(test_df.groupby(["district_id"])["gap_diff"].mean())

    def submission_process(self):
        predict_df = pd.read_csv("./train_data/s1_predict_atw.csv", sep=",", header=None, names=["district_id",
                                                                                                 "time",
                                                                                                 "tj_level_1",
                                                                                                 "tj_level_2",
                                                                                                 "tj_level_3",
                                                                                                 "tj_level_4",
                                                                                                 "weather",
                                                                                                 "temperature",
                                                                                                 "pm25"])
        predict_df["week"] = [self.get_weekday(item) for item in predict_df["time"]]
        good_df = predict_df.copy()
        print "good_df.shape", good_df.shape

        predict_df["time"] = [self.g_time(item) for item in predict_df["time"]]
        ## 按照地区和时间进行一个排序
        predict_df = predict_df.sort_values(by=["district_id", "time"])
        predict_df = predict_df.fillna(method='pad')
        good_df = good_df.fillna(method='pad')
        predictions = []
        gf = []
        ts = []
        mape = []
        for district_id in range(1, 67):
            print "district_id=", district_id
            train_data, train_target_data = self.get_train_data(district_id)
            test_data, test_target_data = self.get_train_data(district_id)

            clf = self.train_process(train_data, train_target_data, test_data, test_target_data)
            prediction = clf.predict(predict_df[predict_df.district_id == district_id])
            for item in prediction:
                predictions.append(item)
            ts.append(self.test_score)
            mape.append(self.mp)
            gf.append(good_df[good_df.district_id == district_id])
        new_good_df = pd.concat(gf)
        print "mape mean:", np.mean(mape)
        self.create_submission(predictions, new_good_df, np.mean(ts))


td = TrainDivide()
td.submission_process()
