# Health Insurance Cost Prediction

A machine learning project that predicts individual health insurance charges based on demographic and health-related features.

---

## 📌 Project Overview

This project builds a regression model using the **Medical Cost Personal Dataset** to estimate insurance charges based on features such as age, BMI, smoking status, and region.

The workflow includes:
- Data preprocessing and encoding
- Model training using Random Forest Regression
- Saving the trained model for reuse
- Running predictions on new datasets

---

## 🧠 Model Details

- Algorithm: Random Forest Regressor
- Library: Scikit-learn
- Target: Insurance charges
- Features: Age, BMI, children, sex, smoker status, region

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Train the model
```bash
python3 src/train.py
```
This will generate models/model.pkl

### 3. Make predictions
import cleaned csv file to data folder and rename 'new_data.csv'
```bash
python3 src/predict.py
```
This will output data/predicted_charges.csv

