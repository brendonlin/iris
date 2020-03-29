import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

data = datasets.load_wine()

X = data.get("data")
y = data.get("target")
feature_names = data.get("feature_names")

df = pd.DataFrame(X, columns=feature_names)

columnsTrans = ColumnTransformer(
    [
        ("std", StandardScaler(), ["alcohol"]),
        ("quant", QuantileTransformer(n_quantiles=100), ["malic_acid"]),
    ],
    remainder="passthrough",
)

result = columnsTrans.fit_transform(df)
print(result)
