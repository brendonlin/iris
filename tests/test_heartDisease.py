import pandas as pd
from iris import heartDisease
from sklearn import model_selection
from sklearn import pipeline
from sklearn import metrics


def getData():
    path = "data/heart.csv"
    df = pd.read_csv(path)
    df = df.sample(df.shape[0], random_state=999)
    Xdata = df.drop("target", axis=1)
    ydata = df["target"]
    return Xdata, ydata


def test_cv():
    doctor = heartDisease.HeartDiseaseDoctor()
    Xdata, ydata = getData()
    X = doctor.readXdata(Xdata)
    y = doctor.readydata(ydata)
    transformer = doctor.getTransformer()
    model = doctor.getModel()
    pipe = pipeline.Pipeline([("transformer", transformer), ("model", model)])
    scores = model_selection.cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    scoreMean, scoreStd = scores.mean(), scores.std()
    print("\nCross Validtion Report")
    print(f"Baseline Score:{scoreMean:.2f} +/-{scoreStd*1:.2f}")


# @pytest.mark.skip
def test_gridsearch():
    doctor = heartDisease.HeartDiseaseDoctor()
    Xdata, ydata = getData()
    X = doctor.readXdata(Xdata)
    y = doctor.readydata(ydata)
    pipe = doctor.getPipe()
    paramGrid = {
        "model__max_depth": [5, 10],
    }
    gscv = model_selection.GridSearchCV(pipe, paramGrid, cv=5, scoring="roc_auc")
    gscv.fit(X, y)
    print("\nGrid Search Report")
    for key in ["mean_test_score", "std_test_score", "rank_test_score"]:
        print(f"{key}:{[round(x,2) for x in gscv.cv_results_[key]]}")
    print(gscv.best_params_)


def test_onetest():
    doctor = heartDisease.HeartDiseaseDoctor()
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
    tyPredictQuant = model.predict_proba(txTranformed)[:, 1]
    print(metrics.confusion_matrix(ty, tyPredict))
    featureImportance = pd.Series(
        dict(zip(doctor.Xdata.columns, model.feature_importances_))
    )
    featureImportance.sort_values(inplace=True)
    print(featureImportance)
    fpr, tpr, thresholds = metrics.roc_curve(ty, tyPredictQuant)
    print(f"AUC Score:{metrics.auc(fpr, tpr)}")
    print(f"ROC AUC Score:{metrics.roc_auc_score(ty, tyPredictQuant)}")
    result = doctor.getPredictDetail(tx, tyPredict, ty, doctor.featureNames)
    result.to_csv("tests/data/hdresult.csv", index=False)


def test_modeling():
    Xdata, ydata = getData()
    doctor = heartDisease.HeartDiseaseDoctor()
    doctor.modeling(Xdata, ydata)
    result = doctor.predict(Xdata)
    print(result)
