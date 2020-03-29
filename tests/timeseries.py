import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dtindex = pd.date_range("2020-01-01", "2020-02-29")
ser = pd.Series(np.random.randint(80, 100, len(dtindex)), index=dtindex)

weekImpact = {
    0: 0.7,
    1: 0.3,
    2: 0.2,
    3: 0.3,
    4: 0.2,
    5: 0.4,
    6: 0.8,
}

serc = ser.copy()
weekdays = []
for i, value in enumerate(ser.values):
    weekday = i % 7
    weekdays.append(weekday)
    impact = weekImpact[weekday]
    serc.iloc[i] = value * impact
arser = serc.rolling(7).mean()
kk = pd.Series((ser / arser).values, index=weekdays)
sesonality = kk.reset_index().groupby("index").mean()
print(sesonality)
# plt.plot(serc)
# plt.plot(arser)
# plt.show()

arr = serc.values
records = {}
for i in range(1, len(arr) - 1):
    coef = np.corrcoef(arr[: (-1 * i)], arr[i:])[0, 1]
    records[i] = coef

# `import statsmodels.api as sm

# sm.graphics.tsa.plot_acf(arr, lags=50)
# plt.show()

from statsmodels import tsa

model = tsa.arima_model.ARMA(serc, (2, 0))
model
