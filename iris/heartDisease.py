#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# from sklearn import naive_bayes
# from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import compose

import joblib


class HeartDiseaseDoctor(object):
    featureMap = {
        "age": "age",
        "sex": "sex",
        "cp": "chest_pain_type",
        "trestbps": "blood_pressure",
        "chol": "cholestoral",
        "fbs": "blood_sugar",
        "restecg": "rest_ecg",
        "thalach": "max_heart_rate",
        "exang": "exercise_induced_angina",
        "oldpeak": "st_depression",
        "slope": "st_slope",
        "ca": "vessels_number",
        "thal": "thalassemia",
    }
    className = "target"
    savePath = "data/heart.joblib"

    def __init__(self):
        self.featureNames = []
        self.Xdata = None

    def readXdata(self, Xdata):
        Xdata_ = Xdata.copy()
        Xdata_.rename(self.featureMap, axis=1, inplace=True)
        Xdata_.replace(
            {
                "sex": {0: "female", 1: "male"},
                "chest_pain_type": {
                    0: "typical angina",
                    1: "atypical angina",
                    2: "non-anginal pain",
                    3: "asymptomatic",
                },
                "blood_sugar": {0: "lower than 120mg/ml", 1: "greater than 120mg/ml"},
                "rest_ecg": {
                    1: "normal",
                    2: "ST-T wave abnormality",
                    3: "left ventricular hypertrophy",
                },
                "exercise_induced_angina": {0: "no", 1: "yes"},
                "st_slope": {0: "upsloping", 1: "flat", 2: "downsloping"},
                "thalassemia": {1: "normal", 2: "fixed defect", 3: "reversable defect"},
            },
            inplace=True,
        )
        Xdata_ = pd.get_dummies(Xdata_, drop_first=True)
        self.Xdata = Xdata_
        self.featureNames = list(Xdata_.columns)
        X = Xdata_.values
        return X

    def readydata(self, ydata):
        y = ydata.values.flatten()
        return y

    def readData(self, Xdata, ydata):
        X = self.readXdata(Xdata)
        y = self.readydata(ydata)
        return (X, y)

    def _getIndex(self, target, inverse=False):
        if inverse:
            return [
                self.featureNames.index(x) for x in self.featureNames if x not in target
            ]
        else:
            return [self.featureNames.index(x) for x in target]

    def getTransformer(self, **params):
        # numvars = ["blood_pressure", "cholestoral", "max_heart_rate", "age"]
        # cateVars = ["cp", "thal"]
        ct = compose.ColumnTransformer(
            [
                # ("norm", preprocessing.StandardScaler(), self._getIndex(numvars)),
                # (
                #     "cate",
                #     preprocessing.OneHotEncoder(handle_unknown="ignore"),
                #     self.getIndex(cateVars),
                # ),
            ],
            remainder="passthrough",
        )
        transformer = pipeline.Pipeline([("norm", ct)])
        transformer.set_params(**params)
        return transformer

    def getModel(self, **params):
        model = ensemble.RandomForestClassifier(max_depth=5)
        model.set_params(**params)
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
        df = df.assign(
            diff=df[self.className] != df[yPrecitColumnName],
            fp=(df[self.className] == 0.0) & (df[yPrecitColumnName] == 1.0),
            fn=(df[self.className] == 1.0) & (df[yPrecitColumnName] == 0.0),
        )
        return df

    def modeling(self, Xdata, ydata):
        print(f"\nStart Modeling")
        X, y = self.readData(Xdata, ydata)
        pipe = self.getPipe()
        pipe.fit(X, y)
        self.fittedPipe = pipe
        joblib.dump(pipe, self.savePath)
        print(f"Model saved at {self.savePath}")
        return pipe

    def predict(self, Xdata):
        X = self.readXdata(Xdata)
        try:
            pipe = joblib.load(self.savePath)
            print(f"Model read from {self.savePath}")
        except Exception as e:
            print(e)
        else:
            result = pipe.predict(X)
            return result
