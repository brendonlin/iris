from collections import Counter
import pandas as pd
from iris import strategyGame
from sklearn import model_selection
from sklearn import pipeline
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
from sklearn import dummy
from sklearn import decomposition

from iris.common import ordinalClassifier
from iris.common import preprocess


def getData(minRatingCount=100):
    path = "tests/data/appstore_games.csv"
    df = pd.read_csv(path)
    df = df[~pd.isnull(df["Average User Rating"])]
    df = df[df["User Rating Count"] >= minRatingCount]
    # df.loc[df["Average User Rating"] <= 2.5, "Average User Rating"] = 2.5
    df.loc[:, "Average User Rating"] = df["Average User Rating"] >= 4.5
    df = df.sample(df.shape[0])
    # df = df.sample(10000)
    df.reset_index(drop=True, inplace=True)
    print(df.columns)
    targetName = "Average User Rating"
    Xdata = df.drop(targetName, axis=1)
    ydata = df[targetName]
    ydata.name = "target"
    ydata = ydata.astype("object")
    return Xdata, ydata


def test_readData():
    Xdata, ydata = getData()
    doctor = strategyGame.strategyGameDoctor()
    X, y = doctor.readData(Xdata, ydata)
    X, y = preprocess.balanceSample(X, y)
    path = "tests/data/gamesResult.csv"
    doctor.Xdata.join(doctor.ydata).to_csv(path, index=False)


def test_cv():
    doctor = strategyGame.strategyGameDoctor()
    Xdata, ydata = getData()
    X = doctor.readXdata(Xdata)
    y = doctor.readydata(ydata)
    # X, y = preprocess.balanceSample(X, y)
    transformer = doctor.getTransformer()
    model = doctor.getModel()
    pipe = pipeline.Pipeline([("transformer", transformer), ("model", model)])
    scores = model_selection.cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    scoreMean, scoreStd = scores.mean(), scores.std()
    print("\nCross Validtion Report")
    print(f"Baseline Score:{scoreMean:.2f} +/-{scoreStd*1:.2f}")


def test_gridsearch():
    doctor = strategyGame.strategyGameDoctor()
    ratingCounts = [10, 30, 50, 80, 100, 150, 200, 500, 1000, 2000, 3000]
    ratingCounts = [100]
    scores = []
    for minRatingCount in ratingCounts:
        Xdata, ydata = getData(minRatingCount)
        X = doctor.readXdata(Xdata)
        y = doctor.readydata(ydata)
        # X, y = preprocess.balanceSample(X, y)
        pipe = doctor.getPipe()
        model1 = ordinalClassifier.OrdinalClassifier(linear_model.LogisticRegression())
        model2 = tree.DecisionTreeClassifier(max_depth=5)
        model3 = ensemble.RandomForestClassifier(max_depth=10)
        model4 = dummy.DummyClassifier(strategy="most_frequent")

        transformer1 = pipeline.Pipeline([("pac", decomposition.PCA())])
        transformer2 = "passthrough"
        # model5 = naive_bayes.GaussianNB()
        paramGrid = {
            "tranformer": [transformer2],
            "model": [model3, model4],
            # "model__max_depth": [3, 5, 7, 10, 15],
        }
        gscv = model_selection.GridSearchCV(pipe, paramGrid, cv=5, scoring="accuracy")
        gscv.fit(X, y)
        print("\nGrid Search Report")
        for key in ["mean_test_score", "std_test_score", "rank_test_score"]:
            print(f"{key}:{[round(x,2) for x in gscv.cv_results_[key]]}")
        scores.append((minRatingCount, list(gscv.cv_results_["mean_test_score"])))
    print(scores)
    # print(gscv.best_params_)


def test_onetest():
    doctor = strategyGame.strategyGameDoctor()
    Xdata, ydata = getData()
    X = doctor.readXdata(Xdata)
    y = doctor.readydata(ydata)
    # X, y = preprocess.balanceSample(X, y)
    rx, tx, ry, ty = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    # rx, ry = preprocess.balanceSample(rx, ry)
    # tx, ty = preprocess.balanceSample(tx, ty)
    transformer = doctor.getTransformer()
    transformer.fit(rx)

    model = doctor.getModel()
    model.fit(transformer.transform(rx), ry)
    txTranformed = transformer.transform(tx)
    tyPredict = model.predict(txTranformed)
    # tyPredictQuant = model.predict_proba(txTranformed)[:, 1]
    # print(metrics.confusion_matrix(ty, tyPredict))
    score = metrics.accuracy_score(ty, tyPredict)
    baseline = max(Counter(ty).values()) / len(ty)
    featureImportance = pd.Series(
        dict(zip(doctor.Xdata.columns, model.feature_importances_))
    )
    featureImportance.sort_values(inplace=True)
    print(featureImportance)
    print(f"Accuracy:{score:.2f}")
    print(f"Baseline:{baseline:.2f}")
    reverseTy = doctor.ylabelEncoder.inverse_transform(ty)
    reversetyPredict = doctor.ylabelEncoder.inverse_transform(tyPredict)
    result = doctor.getPredictDetail(
        tx, reverseTy, reversetyPredict, doctor.featureNames
    )

    result.to_csv("tests/data/sgresult.csv", index=False)
