#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt

# from sklearn import tree
import pandas as pd
import numpy as np
from sklearn import linear_model

# from sklearn import naive_bayes
# from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import compose

# from sklearn import metrics


# import altair as alt


# class HDTranformer(preprocessing.)
class HeartDiseaseDoctor(object):
    def __init__(self):
        self.featureNames = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]

        self.className = "target"
        self.transformedFeatureNames = []

    def readData(self, df):
        Xdata = df[self.featureNames]
        ydata = df[self.className]
        # Xdata["thalach_lager_140"] = (Xdata["thalach"] > 140) * 1
        # self.featureNames += ["thalach_lager_140"]
        # Xdata["oldpeak_small"] = (Xdata["oldpeak"] <= 0.5) * 1
        # self.featureNames += ["oldpeak_small"]
        X = Xdata.values
        y = ydata.values.flatten()
        return X, y

    def getIndex(self, target, inverse=False):
        print(self.featureNames)
        if inverse:
            return [
                self.featureNames.index(x) for x in self.featureNames if x not in target
            ]
        else:
            return [self.featureNames.index(x) for x in target]

    def getTransformer(self, **params):
        numvars = ["trestbps", "chol", "thalach", "age"]
        cateVars = ["cp", "thal"]
        ct = compose.ColumnTransformer(
            [
                ("norm", preprocessing.StandardScaler(), self.getIndex(numvars)),
                (
                    "cate",
                    preprocessing.OneHotEncoder(handle_unknown="ignore"),
                    self.getIndex(cateVars),
                ),
            ],
            remainder="passthrough",
        )
        pipe = pipeline.Pipeline([("norm", ct)])
        pipe.set_params(**params)
        return pipe

    def getModel(self, **params):
        model = linear_model.LogisticRegression(
            max_iter=3000, penalty="l1", solver="liblinear"
        )
        model.set_params(**params)
        return model

    def getFullPipe(self, **params):
        transformer = self.getTransformer()
        model = self.getModel()
        fullPipe = pipeline.Pipeline([("tranformer", transformer), ("model", model)])
        fullPipe.set_params(**params)
        return fullPipe

    def getPredictDetail(self, X, y, yPredict, transformedFeatureNames):
        yPrecitColumnName = f"{self.className}_predict"
        values = np.hstack([X, np.vstack([y, yPredict]).T])
        columns = transformedFeatureNames + [
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


# def main():
#     df = getData()

#     rx, tx, ry, ty = model_selection.train_test_split(X, y, test_size=0.2)

#     pipe.fit(rx, ry)
#     print(metrics.classification_report(pipe.predict(tx), ty))


# gscv = model_selection.GridSearchCV(pipe, paramGrid, cv=5, scoring="f1")
# gscv.fit(rx, ry)

# for key in ["mean_test_score", "std_test_score", "rank_test_score"]:
#     print(f"{key}:{[round(x,2) for x in gscv.cv_results_[key]]}")
# print(gscv.best_params_)


# # pipe.set_params(norm=preprocessing.StandardScaler())
# scores = model_selection.cross_val_score(pipe, rx, ry, cv=5, scoring="f1")
# scoreMean, scoreStd = scores.mean(), scores.std()
# print(f"Baseline Score:{scoreMean:.2f} +/-{scoreStd*1:.2f}")

# getFindex(["cp", "thal"], inverse=True)


# newRx = ct.fit_transform(rx)
# corr = [np.corrcoef(newRx[:, col], ry)[0, 1] for col in range(newRx.shape[1])]
# # ct.categories_
# # ct.get_feature_names()
# newFeatureNames = [x for x in featureNames if x not in ["cp", "thal"]] + list(
#     ct.named_transformers_["cate"].get_feature_names(["cp", "thal"])
# )
# len(corr)
# len(newFeatureNames)


# statdf = pd.DataFrame({"corr": corr, "feature": newFeatureNames})
# statdf

# alt.Chart(statdf).mark_bar().encode(
#     alt.Y("feature:O", sort=None), alt.X("corr:Q"), tooltip="corr:Q"
# )


# # plt.figure(figsize=(20,5))
# axes = pd.DataFrame(newRx, columns=newFeatureNames).hist(figsize=(20, 10))
# plt.tight_layout()


# groupMean = (
#     pd.DataFrame(newRx, columns=newFeatureNames)
#     .groupby(ry)
#     .mean()
#     .stack()
#     .reset_index()
# )
# groupMean.columns = ["target", "feature", "value"]
# alt.Chart(groupMean).mark_bar().encode(x="target:O", y="value",).facet(column="feature")
