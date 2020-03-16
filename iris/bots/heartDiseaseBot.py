from ..modelBot import ModelBot


class HeartDiseaseBot(ModelBot):
    def getData(self):
        uri = "./data/heart.csv"
        data = pd.read_csv(uri)
        return data

    def preprocess(self, data):
        self.featureNames = list(df.columns)
        self.targetName = "target"
        self.featureNames.remove(className)
        X = data[self.featureNames].values
        y = data[slef.targetName].values.flatten()
        return X, y

    def test(self, X, y):
        Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(
            X, y, test_size=0.2
        )
        self.model.train(Xtrain, ytrain)
        report = {}
        ypredict = model.predict(Xtest)
        crossTable = pd.crosstab(
            ytest, ypredict, rownames=["Actual"], colnames=["predict"]
        )
        scores = model_selection.cross_validate(
            self.model, X, y, cv=5, scoring="roc_auc"
        )
        # scoreMean, scoreStd = scores["test_score"].mean(), scores["test_score"].std()
        # print(f"Score:{scoreMean:.2f} +-{scoreStd*1:.2f}")
        report = {"crossTable": crossTable, "scores": scores}
        return report
