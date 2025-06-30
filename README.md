# Advanced Stock Price Prediction Project

This project focuses on building, training, and evaluating sequential models using TensorFlow/Keras for predicting stock price movements based on relative values. The core objective is to compare the performance of an LSTM-only model against an LSTM model enhanced with an attention mechanism.

## Project Objectives

The main goals of this project are:

1. **Data Acquisition:**  
   Download daily stock prices for a chosen stock (e.g., SAP.DE as shown in the notebook) over the last 10 years using the `yfinance` package.

2. **Data Preparation:**  
   Implement a dataset generation function using a sliding window approach.  
   - **Input Features:** Relative values (e.g., percentage changes or log returns) of 50 consecutive days of daily stock prices. The window slides one day at a time.  
   - **Target (Ground Truth):** The relative change of the stock price on the following day.

3. **Model Implementation:**  
   - **Model 1 (LSTM-only):** Develop a Recurrent Neural Network using LSTM or GRU cells to predict the relative stock price change. Experiment with different architectures and hyperparameters (number of cells/layers, dropout rates, learning rates). Bidirectional RNNs are an optional exploration.  
   - **Model 2 (LSTM with Attention Mechanism):** Implement an LSTM-based recurrent neural network incorporating an attention mechanism, following an external article's guidelines.

4. **Training:**  
   Train both models on the first 85% of the data using appropriate loss functions and optimizers. Use techniques like Dropout, Early Stopping, and Regularization to prevent overfitting, ensuring comparable training conditions for fair evaluation.

5. **Evaluation and Comparison:**  
   Evaluate model performance on the last 15% of the dataset (test set) through autoregressive prediction.  
   - **Prediction Process:** The first prediction uses the initial `window_size` data points, and subsequent predictions include the most recent predictions in a sliding window approach.  
   - **Metrics:** Employ suitable evaluation metrics for both models.  
   - **Visualization:** Visualize predicted relative price changes against actual changes and reconstruct/plot absolute prices by applying cumulative relative changes.

6. **Experimentation:**  
   Repeat or automate the process for different sliding window sizes (e.g., 10, 25, 50, 100 days). Analyze and discuss the effect of window size on learning temporal patterns.

---

## Implementation Details (Based on TeamNr7_Advanced_Stock_Price_Prediction.ipynb)

The Jupyter Notebook `TeamNr7_Advanced_Stock_Price_Prediction.ipynb` provides the full implementation for this project.

### 1. Data Acquisition  
- **Library Used:** `yfinance`  
- **Stock Symbol:** `SAP.DE`  
- **Date Range:** Data downloaded from 2015-04-30 to 2025-04-30

### 2. Data Preparation  
- Uses `pandas`, `numpy`, and `sklearn.preprocessing.MinMaxScaler` for data manipulation and scaling  
- Functions for calculating relative changes and normalizing data  
- Splits data into training (85%) and testing (15%) sets

### 3. Model Implementation  
- Utilizes `tensorflow.keras.models.Sequential` and layers such as `LSTM`, `GRU`, `Dense`, `Dropout`, `Bidirectional`, `AdditiveAttention`, `Permute`, `Reshape`, and `Multiply`  
- **Model 1 (LSTM-only):** A sequential model with LSTM and Dense layers  
- **Model 2 (LSTM with Attention):** An LSTM model incorporating `AdditiveAttention` for weighting sequences

### 4. Training and Evaluation  
- Uses `Adam` optimizer and appropriate loss functions  
- Includes model compilation, training, evaluation, and plotting actual vs. predicted values

### 5. Experimentation with Sliding Window Sizes  
- Explicitly addresses different window sizes (10, 25, 50, 100 days)  
- Discusses trends and patterns in model performance metrics

### 6. Discussion and Future Improvements  
- Suggestions include hyperparameter tuning, regularization, data augmentation, and ensemble modeling

---

---

## References

- Yahoo Finance: [https://finance.yahoo.com/](https://finance.yahoo.com/)
- yfinance Python package: [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)
- Attention Mechanism article: *[Insert full link or citation here]*  
- TensorFlow tutorials and documentation: [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)  
- TensorFlow Additive Attention API: [https://www.tensorflow.org/api_docs/python/tf/keras/layers/AdditiveAttention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AdditiveAttention)

---

Feel free to reach out if you have any questions or suggestions!

