# Time_series_LSTM_Autoencoder

# Time Series Imputation using LSTM Autoencoder

This repository explores the use of LSTM Autoencoders for robust time series imputation and outlier correction. It contains two notebooks:

### 🔹 Temperature_Autoencoder_LSTM_main.ipynb
- Trains an LSTM autoencoder on weather data (`temperature_2m`)
- Introduces synthetic outliers
- Compares imputation strategies:
  - Simple mean
  - Simple median
  - LSTM-AE based reconstruction
- Includes RMSE evaluation and visual plots

### 🔹 GE_Outlier_prediction__from_DXB_Temp_model_v2.ipynb
- Applies the trained LSTM Autoencoder to GE stock price data
- Evaluates performance of different imputation techniques
- Adapts architecture from the temperature model
- Fine tuning to improve performance <TBD>

## 🧠 LSTM Autoencoder Architecture

```python
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(sequence_length, 1), return_sequences=False))
model.add(Dropout(0.2))
model.add(RepeatVector(sequence_length))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))


<img width="582" alt="image" src="https://github.com/user-attachments/assets/3fe94a1c-7a50-44cf-9127-51cc4417cb9d" />

