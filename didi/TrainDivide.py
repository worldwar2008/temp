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
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV



class TrainDivide(object):
    def __init__(self):
        self.mape_loss = make_scorer(self.my_mape_loss_func, greater_is_better=False)
        self.est = GradientBoostingRegressor(loss='huber', n_estimators=3500, max_depth=6, max_features=6,
                                             learning_rate=0.03)
        self.rfr = RandomForestRegressor(n_estimators=10, n_jobs=1, min_samples_split=3, max_features=5)
        self.test_score = 0
        self.mp = 0
        self.param_grid = {'learning_rate': [0.01, 0.02, 0.05, 0.1],
                           'max_depth': [6, 8,10],
                           'max_features': [0.1, 0.5, 1.0]}

    @staticmethod
    def g_time(time_str):
        return int(time_str.split("-")[-1])

    @staticmethod
    def time_index(time_str):
        m = int(time_str.split("-")[-1])
        d = int(time_str.split("-")[-2])
        return d * 144 + m

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
        train_df["time_index"] = [self.time_index(item) for item in train_df["time"]]
        train_df["time"] = [self.g_time(item) for item in train_df["time"]]
        # train_df = train_df.sort_values(by=["district_id", "time"])
        train_df = train_df.fillna(method='pad')
        train_df = train_df.fillna(method='backfill')
        # train_df = train_df.dropna()

        train_df = train_df[train_df.district_id == district_id]
        train_df = train_df.sort_values(by=["district_id", "time_index"])
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
        test_df["time_index"] = [self.time_index(item) for item in test_df["time"]]
        test_df["time"] = [self.g_time(item) for item in test_df["time"]]
        # test_df = test_df.sort_values(by=["district_id", "time"])
        test_df = test_df.fillna(method='pad')
        test_df = test_df.fillna(method='backfill')
        # test_df = test_df.dropna()
        test_df = test_df[test_df.district_id == district_id]
        test_df = test_df.sort_values(by=["district_id", "time_index"])
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
        val_scores = cross_val_score(self.est, train_data, train_target_data, cv=2, n_jobs=1, scoring=mape_loss)
        print "val_scores", val_scores
        print "val_scores_mean", np.mean(val_scores)
        return np.mean(val_scores)

    def train_process(self, train_data, train_target_data, test_data, test_target_data):
        # clf = rfr.fit(train_data, train_target_data)
        clf = self.est.fit(train_data, train_target_data)
        ############################################################################
        # test_dev, ax = self.deviance_plot(self.est, test_data, test_target_data)
        # ax.legend(loc='upper right')
        # # add some annotations
        # ax.annotate('Lowest test error', xy=(test_dev.argmin() + 1, test_dev.min() + 0.02), xycoords='data',
        #             xytext=(150, 1.0), textcoords='data',
        #             arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
        #             )
        # plt.show()



        # ann = ax.annotate('', xy=(800, test_dev[799]),  xycoords='data',
        #                   xytext=(800, est.train_score_[799]), textcoords='data',
        #                   arrowprops=dict(arrowstyle="<->"))
        # ax.text(810, 0.25, 'train-test gap')
        #############################################################################

        print "train_score:", np.mean(self.est.train_score_)
        test_predict = clf.predict(test_data)
        # 计算loss 和 score的参数不同,请注意
        self.test_score = clf.loss_(test_predict, test_target_data)

        print "test_score: ", self.test_score
        test_data["predict"] = test_predict
        test_data["gap_diff"] = np.abs(test_data["predict"].round(0) - test_target_data) / (test_target_data + 1)
        test_df = test_data.fillna("pad")
        mape = np.mean(test_df.groupby(["district_id"])["gap_diff"].mean())
        self.mp = mape
        print "mape_socre: ", mape

        # # Plot feature importance
        # feature_importance = clf.feature_importances_
        # # make importances relative to max importance
        # feature_importance = 100.0 * (feature_importance / feature_importance.max())
        # sorted_idx = np.argsort(feature_importance)
        # pos = np.arange(sorted_idx.shape[0]) + .5
        # plt.subplot(1, 2, 2)
        # plt.barh(pos, feature_importance[sorted_idx], align='center')
        # plt.yticks(pos, train_data.feature_names[sorted_idx])
        # plt.xlabel('Relative Importance')
        # plt.title('Variable Importance')
        # plt.show()
        return clf

    def deviance_plot(self, est, X_test, y_test, ax=None, label='', train_color='#2c7bb6',
                      test_color='#d7191c', alpha=1.0):
        """Deviance plot for ``est``, use ``X_test`` and ``y_test`` for test error. """
        n_estimators = len(self.est.estimators_)
        test_dev = np.empty(n_estimators)

        for i, pred in enumerate(est.staged_predict(X_test)):
            test_dev[i] = est.loss_(y_test, pred)

        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = plt.gca()

        ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label,
                linewidth=2, alpha=alpha)
        ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color,
                label='Train %s' % label, linewidth=2, alpha=alpha)
        ax.set_ylabel('Error')
        ax.set_xlabel('n_estimators')
        # ax.set_ylim((0, 10))
        return test_dev, ax

    def my_mape_loss_func(self, ground_truth, predictions):
        diff = np.mean(np.abs(ground_truth - predictions.round(0)) / np.abs(ground_truth + 1e-9))
        # print "guo_diff:", diff
        return np.abs(diff)

    def my_custom_loss_func(self, ground_truth, predictions):
        diff = np.abs(ground_truth - predictions).max()
        return np.log(1 + diff)

    def get_mape(self, test_data, test_target_data):
        test_predict = self.clf.predict(test_data)
        test_data["predict"] = test_predict
        test_data["gap_diff"] = np.abs(test_data["predict"] - test_target_data) / (test_target_data + 1e-2)
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
        predict_df["time_index"] = [self.time_index(item) for item in predict_df["time"]]
        predict_df["time"] = [self.g_time(item) for item in predict_df["time"]]
        predict_df = predict_df.sort_values(by=["district_id", "time_index"])
        predict_df = predict_df.loc[:, ["district_id",
                                        "time",
                                        "week",
                                        "tj_level_1",
                                        "tj_level_2",
                                        "tj_level_3",
                                        "tj_level_4",
                                        "weather",
                                        "temperature",
                                        "pm25"]]
        ## 按照地区和时间进行一个排序

        predict_df = predict_df.fillna(method='pad')
        good_df = good_df.fillna(method='pad')
        predictions = []
        gf = []
        ts = []
        mape = []
        holdout = []
        for district_id in range(1, 67):
            canshu = 0.85
            print "district_id=", district_id
            train_data, train_target_data = self.get_train_data(district_id)
            test_data, test_target_data = self.get_test_data(district_id)
            ll = len(train_data)
            good_train_data = train_data[0:int(ll * canshu)]
            good_train_target_data = train_target_data[0:int(ll * canshu)]
            good_test_data = train_data[int(ll * (1 - canshu)):]
            good_test_target_data = train_target_data[int(ll * (1 - canshu)):]

            clf = self.train_process(good_train_data, good_train_target_data, good_test_data, good_test_target_data)
            prediction = clf.predict(predict_df[predict_df.district_id == district_id])
            for item in prediction:
                predictions.append(item)
            ts.append(self.test_score)
            mape.append(self.mp)
            gf.append(good_df[good_df.district_id == district_id])
            #### holdout result
            holdout_value = np.mean(np.abs(self.est.predict(test_data).round(0) - test_target_data) / (test_target_data + 1))
            print "holdout mean:", holdout_value
            holdout.append(holdout_value)
        new_good_df = pd.concat(gf)
        print "mape mean:", np.mean(mape)
        print "ts mean:", np.mean(ts)
        print "holdout mean:", np.mean(holdout)
        self.create_submission(predictions, new_good_df, np.mean(holdout))

    def val_process(self):
        mape = []
        for district_id in range(1, 67):
            print "district_id=", district_id
            train_data, train_target_data = self.get_train_data(district_id)
            # test_target_data = self.get_train_data(district_id)

            g_tmp = self.cross_val(train_data, train_target_data)
            mape.append(g_tmp)
        print "mape mean: ", np.mean(mape)

    def grid_search(self):
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
        predict_df["time_index"] = [self.time_index(item) for item in predict_df["time"]]
        predict_df["time"] = [self.g_time(item) for item in predict_df["time"]]
        predict_df = predict_df.sort_values(by=["district_id", "time_index"])
        predict_df = predict_df.loc[:, ["district_id",
                                        "time",
                                        "week",
                                        "tj_level_1",
                                        "tj_level_2",
                                        "tj_level_3",
                                        "tj_level_4",
                                        "weather",
                                        "temperature",
                                        "pm25"]]
        ## 按照地区和时间进行一个排序

        predict_df = predict_df.fillna(method='pad')
        good_df = good_df.fillna(method='pad')
        predictions = []
        gf = []
        ts = []
        mape = []
        holdout = []
        for district_id in range(1, 67):
            canshu = 0.85
            print "district_id=", district_id
            train_data, train_target_data = self.get_train_data(district_id)
            test_data, test_target_data = self.get_test_data(district_id)
            ll = len(train_data)
            good_train_data = train_data[0:int(ll * canshu)]
            good_train_target_data = train_target_data[0:int(ll * canshu)]
            good_test_data = train_data[int(ll * (1 - canshu)):]
            good_test_target_data = train_target_data[int(ll * (1 - canshu)):]
            gs_cv = GridSearchCV(self.est, self.param_grid, n_jobs=10).fit(train_data, train_target_data)
            print "gs_cv %s 最优参数:"%district_id, gs_cv.best_params_
            prediction = gs_cv.best_estimator_.predict(predict_df[predict_df.district_id == district_id])


            #clf = self.train_process(good_train_data, good_train_target_data, good_test_data, good_test_target_data)
            #prediction = clf.predict(predict_df[predict_df.district_id == district_id])
            for item in prediction:
                predictions.append(item)
            ts.append(self.test_score)
            mape.append(self.mp)
            gf.append(good_df[good_df.district_id == district_id])
            #### holdout result
            holdout_value = np.mean(np.abs(gs_cv.best_estimator_.predict(test_data).round(0) - test_target_data) / (test_target_data + 1))
            print "holdout mean:", holdout_value
            holdout.append(holdout_value)
        new_good_df = pd.concat(gf)
        print "mape mean:", np.mean(mape)
        print "ts mean:", np.mean(ts)
        print "holdout mean:", np.mean(holdout)
        self.create_submission(predictions, new_good_df, np.mean(holdout))



td = TrainDivide()
#td.submission_process()
td.grid_search()
