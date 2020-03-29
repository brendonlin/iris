import tempfile
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

cachefile = tempfile.mktemp()

bostonData = datasets.load_boston()

X = bostonData.get("data")
y = bostonData.get("target")

xr, xt, yr, yt = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline(
    [("normlize", preprocessing.StandardScaler()), ("model", RandomForestRegressor())],
    memory=cachefile,
)

pipeline.set_params(model__min_samples_split=10)
pipeline.fit(xr, yr)
result = pipeline.predict(xr)
pipeline.score(xt, yt)
