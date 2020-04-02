import re
import pandas as pd
import numpy as np
from sklearn import compose
from sklearn import tree
from sklearn import pipeline
from sklearn import ensemble
from sklearn import preprocessing

from .common import ordinalClassifier

# from sklearn import preprocessing


def toFeatureMap(featureNames):
    def parse(x):
        return re.sub(r"[\s\-]+", "_", x.lower())

    fmap = {x: parse(x) for x in featureNames}
    return fmap


class strategyGameDoctor(object):
    originFeatureNames = [
        "URL",
        "ID",
        "Name",
        "Subtitle",
        "Icon URL",
        "Price",
        "In-app Purchases",
        "Description",
        "Developer",
        "Age Rating",
        "Languages",
        "Size",
        "Primary Genre",
        "Genres",
        "Original Release Date",
        "Current Version Release Date",
        "User Rating Count",
    ]
    featureMap = toFeatureMap(originFeatureNames)
    featureMap.update(
        {
            "Original Release Date": "original_dt",
            "Current Version Release Date": "cur_version_dt",
        }
    )

    def __init__(self):
        self.Xdata = None
        self.ydata = None
        self.className = "target"
        self.ylabelEncoder = preprocessing.LabelEncoder()

    def readData(self, Xdata, ydata):
        X = self.readXdata(Xdata)
        y = self.readydata(ydata)
        return (X, y)

    def readydata(self, ydata):
        self.ydata = ydata
        y = ydata.values.flatten()
        self.ylabelEncoder.fit(y)
        y = self.ylabelEncoder.transform(y)
        return y

    def readXdata(self, Xdata):
        xd = Xdata[list(self.featureMap.keys())].copy()
        # xd.reset_index(inplace=True)
        xd.rename(self.featureMap, axis=1, inplace=True)
        xd.drop(
            [
                "url",
                "id",
                "primary_genre",
                "developer",
                # "user_rating_count"
            ],
            axis=1,
            inplace=True,
        )

        xd = xd.assign(
            is_with_subtitle=pd.isnull(xd["subtitle"]),
            price_zone=pd.cut(
                xd["price"],
                [0, 0.01, 4.99, 9.99, 19.99, 59.99, float("inf")],
                include_lowest=True,
            ),
            is_iap=~pd.isnull(xd["in_app_purchases"]),
            log_size=(xd["size"] / 1024 / 1024).apply(lambda x: np.log(x)),
            cur_release_duration=datediff(xd["cur_version_dt"], xd["original_dt"]),
        )

        xd = xd.join(processIAP(xd["in_app_purchases"]))
        xd = xd.join(processLanguages(xd["languages"]))

        xd = xd.join(processGenres(xd["genres"]))
        xd = xd.join(processDate(xd["original_dt"], "original_dt"))

        xd = xd.join(processDate(xd["cur_version_dt"], "cur_version_dt"))
        xd = pd.get_dummies(xd, columns=["age_rating", "price_zone"])
        xd.drop(
            [
                "name",
                "description",
                "size",
                "cur_version_dt",
                "original_dt",
                "icon_url",
                "in_app_purchases",
                "subtitle",
                "languages",
                "genres",
                "price",
            ],
            axis=1,
            inplace=True,
        )
        self.Xdata = xd
        self.featureNames = list(xd.columns)
        X = xd.values
        return X

    def getTransformer(self, **params):

        ct = compose.ColumnTransformer([], remainder="passthrough",)
        transformer = pipeline.Pipeline([("norm", ct)])
        transformer.set_params(**params)
        return transformer

    def getModel(self, **params):
        # model = ordinalClassifier.OrdinalClassifier(
        #     ensemble.RandomForestClassifier(max_depth=5)
        # )
        model = ordinalClassifier.OrdinalClassifier(
            tree.DecisionTreeClassifier(max_depth=5)
        )
        # model = tree.DecisionTreeClassifier(max_depth=5)
        # model.set_params(**params)
        return model

    def getPipe(self, **params):
        transformer = self.getTransformer()
        model = self.getModel()
        pipe = pipeline.Pipeline([("tranformer", transformer), ("model", model)])
        pipe.set_params(**params)
        return pipe

    def getPredictDetail(self, X, y, yPredict, featureNames):
        yPrecitColumnName = f"{self.className}_predict"
        values = np.hstack([X, np.vstack([y, yPredict]).T])
        columns = featureNames + [
            self.className,
            yPrecitColumnName,
        ]
        df = pd.DataFrame(values, columns=columns)
        # df = df.assign(
        #     diff=df[self.className] != df[yPrecitColumnName],
        #     fp=(df[self.className] == 0.0) & (df[yPrecitColumnName] == 1.0),
        #     fn=(df[self.className] == 1.0) & (df[yPrecitColumnName] == 0.0),
        # )
        return df


def processIAP(iapdata):
    records = []
    for value in iapdata.values:
        if value is np.nan:
            min_iap, mediumn_iap, max_iap, n_iap = (0, 0, 0, 0)
        else:
            numbers = [float(x) for x in str(value).split(", ")]
            min_iap, mediumn_iap, max_iap = np.percentile(numbers, [0, 50, 100])
            n_iap = len(numbers)
        record = (min_iap, mediumn_iap, max_iap, n_iap)
        records.append(record)

    df = pd.DataFrame(
        np.array(records), columns=["min_iap", "mediumn_iap", "max_iap", "n_iap"]
    )
    return df


def processLanguages(landata):
    columns = ["nlan", "only_en", "with_zh", "with_ja", "multi_lan"]
    records = []
    for value in landata.values:
        languages = str(value).split(", ")
        nlan = len(languages)
        with_zh = "ZH" in languages
        with_ja = "JA" in languages
        only_en = nlan == 1 and "EN" in languages
        multi_lan = nlan > 3
        record = (nlan, only_en, with_zh, with_ja, multi_lan)
        records.append(record)
    df = pd.DataFrame(records, columns=columns)
    return df


def processGenres(genredata):
    k = genredata.str.split(", ", expand=True).stack().reset_index()
    k.columns = ["index", "rn", "genre"]
    df = pd.crosstab(k["index"], k["genre"])
    df.drop(["Games", "Strategy"], axis=1, inplace=True)
    df.columns = [f"genre_{x}" for x in df.columns]
    # df.reset_index(inplace=True)
    return df


def processDate(datedata, prefix=""):
    dt = pd.to_datetime(datedata)
    df = pd.concat([2020 - dt.dt.year, dt.dt.month, dt.dt.day], axis=1)
    df.columns = [f"{prefix}_{x}" for x in ["year", "month", "day"]]
    return df


def datediff(x, y):
    durationDays = (pd.to_datetime(x) - pd.to_datetime(y)).dt.days
    durationMonth = durationDays // 31
    return durationMonth
