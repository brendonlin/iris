import pandas as pd
from iris import heartDisease
from sklearn import model_selection
from sklearn import pipeline
from sklearn import metrics

# from sklearn import preprocessing
# from loguru import logger

# import pytest


def getData():
    path = "data/heart.csv"
    df = pd.read_csv(path)
    return df


def test_HeartDiseaseDoctor():
    doctor = heartDisease.HeartDiseaseDoctor()
    df = getData()
    df = df.sample(df.shape[0], random_state=999)
    X, y = doctor.readData(df)
    transformer = doctor.getTransformer()
    model = doctor.getModel()
    pipe = pipeline.Pipeline([("transformer", transformer), ("model", model)])
    rx, tx, ry, ty = model_selection.train_test_split(X, y, test_size=0.3)
    scores = model_selection.cross_val_score(pipe, rx, ry, cv=5, scoring="f1")
    scoreMean, scoreStd = scores.mean(), scores.std()
    print("\nCross Validtion Report")
    print(f"Baseline Score:{scoreMean:.2f} +/-{scoreStd*1:.2f}")


# @pytest.mark.skip
def test_HeartDiseaseDoctor2():
    doctor = heartDisease.HeartDiseaseDoctor()
    df = getData()
    df = df.sample(df.shape[0], sample=999)
    X, y = doctor.readData(df)
    # rx, tx, ry, ty = model_selection.train_test_split(X, y, test_size=0.2)
    pipe = doctor.getFullPipe()
    paramGrid = {
        # "norm": [ct, ],
        "model__penalty": ["l1", "l2"],
    }
    gscv = model_selection.GridSearchCV(pipe, paramGrid, cv=5, scoring="f1", refit=True)
    gscv.fit(X, y)
    print("\nGrid Search Report")
    for key in ["mean_test_score", "std_test_score", "rank_test_score"]:
        print(f"{key}:{[round(x,2) for x in gscv.cv_results_[key]]}")
    print(gscv.best_params_)
    # import ipdb
    # ipdb.set_trace()


def test_train():
    doctor = heartDisease.HeartDiseaseDoctor()
    df = getData()
    df = df.sample(df.shape[0], random_state=999)
    doctor = heartDisease.HeartDiseaseDoctor()
    X, y = doctor.readData(df)
    rx, tx, ry, ty = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    transformer = doctor.getTransformer()
    transformer.fit(rx)
    # cateVars = ["cp", "thal"]
    # cateNames = (
    #     transformer.named_steps["norm"]
    #     .named_transformers_["cate"]
    #     .get_feature_names(cateVars)
    # )
    # transformedFeatureNames = [
    #     x for x in doctor.featureNames if x not in cateVars
    # ] + list(cateNames)
    model = doctor.getModel()
    model.fit(transformer.transform(rx), ry)
    txTranformed = transformer.transform(tx)
    tyPredict = model.predict(txTranformed)
    print(len(doctor.featureNames))
    print(txTranformed.shape[1])
    print(f"Score:{metrics.roc_auc_score(ty, tyPredict)}")
    result = doctor.getPredictDetail(tx, tyPredict, ty, doctor.featureNames)
    result.to_csv("tests/data/hdresult.csv", index=False)
