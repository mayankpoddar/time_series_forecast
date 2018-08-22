# Time Series Forecast using LSTM

### Description
A Time Series Prediction Model on the Internation Air Passengers dataset. <br />
The dataset can be downloaded from here: [international-airline-passengers.csv](https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line) <br />

### Setup Instructions
This code is compatible with: python 3.6.0 <br />
```
pip install -r requirements.txt
```

### Run Instructions
```
python timeSeries.py
```

### Model
- Preprocessing
The dataset is converted to a stationary time series using Log Transformation and Differencing.

- Model
  - LSTM with number of units equal to twice of lookback value (here, 30) with a dropout of 0.1 and tanh activation
  - Fully Connected Dense layer with 1 unit and tanh activation
  
### Performance
![Plot](https://github.com/mayankpoddar/time_series_forecast/blob/master/dataset.png)
- Original Dataset (Blue)
- Fitted Trend (Red)
- Forecast (Green) <br />
Training on 500 epochs yields a loss: 6.6662e-04. The forecast is done on an additional 100 months.
