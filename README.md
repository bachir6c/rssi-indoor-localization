# 📡 Indoor Localization Using Neural Networks

This project uses **RSSI signal strength measurements** from 4 gateways to estimate the **(x, y) position** of an object or user using **neural networks** implemented with TensorFlow/Keras.

---

## 🧠 Objective

Predict the actual position from measured RSSI signals using a deep learning model trained on simulated or real data.

---

## 📁 Data

The data comes from **4 CSV files**:
- `RSSI_0.csv`
- `RSSI_1.csv`
- `RSSI_2.csv`
- `RSSI_3.csv`

Each file contains signal strength values (RSSI) received from one gateway.

---

## 🔧 Pipeline

1. **Data Loading & Visualization** using heatmaps.
2. **Data Cleaning**: removal of the last row and column, with missing `NaN` values filled using the mean.
3. **Dataset Construction**: RSSI values as input features and position `(x, y)` as target output.
4. **Modeling with Keras**:
   - Simple dense neural network architecture `(64-64-2)`
   - Improved model using `Dropout`, `ReduceLROnPlateau`, and additional layers.
5. **Evaluation**:
   - `MSE`, `MAE`, `R²`
   - Training loss curves
   - Localization error histogram

---

## 📦 Libraries

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `keras`
- `scikit-learn`

---

## ⚙️ Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/bachir6c/rssi-indoor-localization.git
cd rssi-indoor-localization
```
