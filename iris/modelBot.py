from abc import ABC


class ModelBot(ABC):

    uri = ""
    featureNames = []
    targetName = ""

    def __init__(self, model):
        self.model = None

    def build(self):
        data = self.getData()
        X, y = self.preprocess()
        report = self.trainTest(X, y)
        return report

    def getData(self):
        pass

    def preprocess(self):
        """
        Feature engineering
        """
        pass

    def trainTest(self):
        pass
