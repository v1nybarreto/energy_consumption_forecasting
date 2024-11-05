
# Energy Consumption Forecasting

This project aims to predict the energy consumption for the next 4 hours using advanced machine learning models. The project is divided into several stages, from data processing to model training and evaluation, all implemented within a single Python script `energy_consumption_forecasting.py`.

## Project Structure

- **consumption_history_2019_2020.csv**: The dataset containing the historical energy consumption from 2019 to 2020.
- **energy_consumption_tuning**: Directory with pre-trained models using CNN-LSTM, GRU, and LSTM architectures.
- **energy_consumption_forecasting.py**: The main script that contains all steps for data processing, sequence creation, model training, evaluation, and forecasting.

## Requirements

To run this project, you need the following Python packages:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `keras`
- `keras_tuner`
- `os`

Install the packages using:
```bash
pip install pandas numpy matplotlib scikit-learn keras keras-tuner
```

## Steps

### 1. Data Loading and Preprocessing

The dataset is loaded and processed to create time-based features such as hour, day, day of the week, month, and year. Outliers are removed using the Interquartile Range (IQR) and Z-Score methods, and the data is normalized using `MinMaxScaler`.

### 2. Time Series Sequence Creation

Sequences of a fixed length (24 hours) are created to capture the time-based patterns in energy consumption. The target is to predict the consumption for the next 4 hours.

### 3. Model Construction

Three models are constructed with adjustable hyperparameters:
- **LSTM**: Long Short-Term Memory model.
- **GRU**: Gated Recurrent Unit model.
- **CNN-LSTM**: A hybrid model combining Convolutional Neural Network and LSTM layers.

### 4. Hyperparameter Tuning

Using `keras_tuner`, each model undergoes hyperparameter tuning to find the best configuration based on validation loss.

### 5. Model Evaluation with Time Series Cross-Validation

The models are evaluated using Time Series Split cross-validation. Three main metrics are calculated for each model:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

### 6. Forecasting

Using the best-performing models, the energy consumption for the next 4 hours is forecasted and visualized.

### 7. Saving and Loading Models

The best models are saved to the `energy_consumption_tuning` directory for future use.

## Usage

Run the following command to execute all steps:
```bash
python energy_consumption_forecasting.py
```

## Results

The results include the forecasted energy consumption values for the next 4 hours, along with plots comparing predictions from each model. The GRU model generally performs best in terms of validation loss and accuracy.

## Directory Structure

```
project-folder/
│
├── energy_consumption_forecasting.py       # Main script with all steps
├── Energy_Consumption_History_2019_2020.csv # Dataset
└── energy_consumption_tuning/
    ├── melhor_modelo_cnn_lstm.h5           # Pre-trained CNN-LSTM model
    ├── melhor_modelo_gru.h5                # Pre-trained GRU model
    └── melhor_modelo_lstm.h5               # Pre-trained LSTM model
```

