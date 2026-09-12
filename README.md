# 💳 Credit Card Fraud Detection — Deep Learning Ensemble

A deep learning system for detecting fraudulent credit card transactions using **MLP, Autoencoder, and Stacking Ensemble** techniques.

The project combines supervised classification with unsupervised anomaly detection to produce a final fraud prediction.
Dataset:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com
## 🚀 Live Demo

[Open the Streamlit App](https://credit-card-fraud-detection-dl-k7cuk4qxhxbpklotlcmurp.streamlit.app/)

---

## 🧠 Project Overview

The system uses three main components:

### 1. MLP — Supervised Classification

A Multi-Layer Perceptron trained to classify transactions as fraudulent or legitimate.

* Dense layers: `128 → 64 → 32 → 16`
* ReLU activation
* Dropout regularization
* Adam optimizer
* SMOTE for class imbalance
* Early stopping
* Fraud probability output

### 2. Autoencoder — Anomaly Detection

The Autoencoder is trained using **legitimate transactions only**.

It learns the normal transaction pattern and uses reconstruction error to identify unusual transactions.

* Gaussian Noise
* Encoder/Decoder architecture
* L2 regularization
* RobustScaler
* Early stopping
* Weighted reconstruction error

### 3. Stacking Ensemble

The MLP probability and Autoencoder anomaly score are combined using a **Logistic Regression meta-model**.

```text
MLP Probability
       +
Autoencoder Score
       ↓
Meta Scaler
       ↓
Logistic Regression
       ↓
Final Fraud Probability
       ↓
FRAUD / LEGITIMATE
```

---

## ⚙️ Preprocessing

The dataset preprocessing includes:

* Duplicate removal
* Converting `Time` into `Hour`
* Log transformation of `Amount`
* Stratified train / validation / test split
* StandardScaler for the MLP
* RobustScaler for the Autoencoder
* SMOTE applied only to the MLP training data

### Final Features

The deployed models expect exactly **30 features**:

```text
V1 ... V28
Amount
Hour
```

The exact feature order is stored in:

```text
feature_columns.pkl
```

---

## 🎯 Threshold Optimization

Fraud detection is a highly imbalanced classification problem, so accuracy alone is not sufficient.

Separate decision thresholds were selected using the validation set:

* **MLP:** threshold optimized for Recall
* **Autoencoder:** threshold optimized using F2-score
* **Ensemble:** threshold optimized using F2-score

This helps the system focus on detecting fraudulent transactions.

---

## 🖥️ Streamlit Application

The deployed application allows users to enter:

* Transaction Amount
* Transaction Hour
* `V1`–`V28` anonymized features

The app returns:

* MLP fraud probability
* Autoencoder anomaly score
* Ensemble fraud probability
* Individual model predictions
* Final `FRAUD` / `LEGITIMATE` decision
* Model thresholds

The application performs **inference only**.

It does not retrain the models or require the original dataset.

---

## 📊 Model Pipeline

```text
                    Transaction
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       StandardScaler         RobustScaler
              │                     │
              ▼                     ▼
             MLP              Autoencoder
              │                     │
              ▼                     ▼
       Fraud Probability      Anomaly Score
              │                     │
              └──────────┬──────────┘
                         ▼
                Logistic Regression
                  Meta-Model
                         │
                         ▼
                 Final Prediction
```

---

## 🛠️ Technologies

* **Python**
* **TensorFlow / Keras**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **imbalanced-learn / SMOTE**
* **Joblib**
* **Streamlit**
* **Git & GitHub**

---

## 📁 Project Structure

```text
credit-card-fraud-detection-dl/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── Credit_Card_Fraud_Full_Pipeline_v7_Tuned__1_.ipynb
│
├── mlp_model.keras
├── autoencoder_model.keras
│
├── scaler.pkl
├── autoencoder_scaler.pkl
├── autoencoder_feature_weights.pkl
│
├── ensemble_meta_scaler.pkl
├── ensemble_meta_model.pkl
│
├── mlp_threshold.pkl
├── autoencoder_threshold.pkl
├── ensemble_threshold.pkl
│
└── feature_columns.pkl
```

> The original `creditcard.csv` dataset is excluded from the repository.

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/sheroukyehia21/credit-card-fraud-detection-dl.git
cd credit-card-fraud-detection-dl
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔒 Data Privacy

The original `creditcard.csv` dataset is **not uploaded to GitHub**.

The Streamlit application uses only the saved:

* Models
* Scalers
* Feature configuration
* Thresholds
* Ensemble artifacts

Therefore, the deployed application does not need access to the original dataset.

---

## 🎓 Project Objective

This project demonstrates a complete Deep Learning workflow for fraud detection:

```text
Data Preprocessing
       ↓
Feature Engineering
       ↓
MLP Training
       ↓
Autoencoder Training
       ↓
Model Stacking
       ↓
Threshold Optimization
       ↓
Model Serialization
       ↓
Streamlit Deployment
```

The project focuses on combining **classification and anomaly detection** into a single deployed fraud detection system.
