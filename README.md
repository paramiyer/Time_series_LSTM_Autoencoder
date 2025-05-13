
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

### 📂 Datasets Used

#### 🌡️ `DubaiTemp.csv`
- Hourly weather data for Dubai starting from January 2010.
- Key feature: `temperature_2m` – temperature measured at 2 meters above ground.
- Used as a **training dataset** to build and validate the LSTM Autoencoder model.
- Perturbations (e.g., replacing random values with zero) are introduced to simulate missing or anomalous readings.

#### 📉 `GE.csv`
- Daily stock price data for General Electric (GE) from May 2023.
- Key feature: `Close` – closing price for the day.
- Used to **test the generalizability** of the trained LSTM Autoencoder on financial time series.
- Demonstrates how an autoencoder trained on environmental data can detect and impute anomalies in a different domain.

## 🧠 LSTM Autoencoder Architecture

```python
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(sequence_length, 1), return_sequences=False))
model.add(Dropout(0.2))
model.add(RepeatVector(sequence_length))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
```

### Final Model Output Visualization

<img width="582" alt="LSTM Autoencoder Output" src="https://github.com/user-attachments/assets/3fe94a1c-7a50-44cf-9127-51cc4417cb9d" />

### Results from using LSTM-AE model for Temperature on missing value imputations on GE Stock (compared with mean & median imputation without fine tuning)

<img width="363" alt="image" src="https://github.com/user-attachments/assets/46021223-e72e-48d5-b600-f0395cc61708" />


### Other uses for this approach
- Generate synthetic sequences from the underlying distribution
- Outlier & Anomaly detection
- Missing Value impuatation

