import pandas as pd
from iris import strategyGame
from sklearn import model_selection
from sklearn import pipeline
from sklearn import metrics
from sklearn import ensemble
from sklearn import tree
from sklearn import dummy

from iris.common import ordinalClassifier
from matplotlib import pyplot as plt


def getData():
    path = "tests/data/appstore_games.csv"
    df = pd.read_csv(path)
    df = df[~pd.isnull(df["Average User Rating"])]
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
    path = "tests/data/gamesResult.csv"
    doctor.Xdata.join(doctor.ydata).to_csv(path, index=False)


def test_cv():
    doctor = strategyGame.strategyGameDoctor()
    Xdata, ydata = getData()
    X = doctor.readXdata(Xdata)
    y = doctor.readydata(ydata)
    transformer = doctor.getTransformer()
    model = doctor.getModel()
    pipe = pipeline.Pipeline([("transformer", transformer), ("model", model)])
    scores = model_selection.cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    scoreMean, scoreStd = scores.mean(), scores.std()
    print("\nCross Validtion Report")
    print(f"Baseline Score:{scoreMean:.2f} +/-{scoreStd*1:.2f}")


def test_gridsearch():
    doctor = strategyGame.strategyGameDoctor()
    Xdata, ydata = getData()
    X = doctor.readXdata(Xdata)
    y = doctor.readydata(ydata)
    pipe = doctor.getPipe()
    model1 = ordinalClassifier.OrdinalClassifier(
        tree.DecisionTreeClassifier(max_depth=5)
    )
    model2 = tree.DecisionTreeClassifier(max_depth=5)
    model3 = ensemble.RandomForestClassifier(max_depth=5)
    model4 = dummy.DummyClassifier(strategy="most_frequent")
    paramGrid = {
        # "model": [model1, model2, model3],
        "model": [model2, model4],
        # "model__max_depth": [3, 5, 7, 10],
    }
    gscv = model_selection.GridSearchCV(pipe, paramGrid, cv=5, scoring="accuracy")
    gscv.fit(X, y)
    print("\nGrid Search Report")
    for key in ["mean_test_score", "std_test_score", "rank_test_score"]:
        print(f"{key}:{[round(x,2) for x in gscv.cv_results_[key]]}")
    print(gscv.best_params_)


def test_onetest():
    doctor = strategyGame.strategyGameDoctor()
    Xdata, ydata = getData()
    X = doctor.readXdata(Xdata)
    y = doctor.readydata(ydata)
    rx, tx, ry, ty = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    transformer = doctor.getTransformer()
    transformer.fit(rx)

    model = doctor.getModel()
    model.fit(transformer.transform(rx), ry)
    txTranformed = transformer.transform(tx)
    tyPredict = model.predict(txTranformed)
    # tyPredictQuant = model.predict_proba(txTranformed)[:, 1]
    # print(metrics.confusion_matrix(ty, tyPredict))

    # featureImportance = pd.Series(
    #     dict(zip(doctor.Xdata.columns, model.feature_importances_))
    # )
    # featureImportance.sort_values(inplace=True)
    # print(featureImportance)
    reverseTy = doctor.ylabelEncoder.inverse_transform(ty)
    reversetyPredict = doctor.ylabelEncoder.inverse_transform(tyPredict)
    result = doctor.getPredictDetail(
        tx, reverseTy, reversetyPredict, doctor.featureNames
    )

    result.to_csv("tests/data/sgresult.csv", index=False)
