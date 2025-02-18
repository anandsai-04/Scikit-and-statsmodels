import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

# Load data
data = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)

# Fit ARIMA model
model = ARIMA(data['value'], order=(p, d, q))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=10)
print(forecast)
