
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("../data/insurance.csv")

df_cleaned = df.dropna()
df_encoded = pd.get_dummies(df_cleaned, columns=[
    "sex",
    "smoker",
    "region",
], drop_first=True)

X = df_encoded.drop("charges", axis=1)
Y = df_encoded["charges"]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=100)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,random_state=42
)
rf.fit(x_train,y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_rf_train_pred))
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_rf_test_pred))
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

result = pd.DataFrame([rf_train_rmse, rf_train_r2, rf_test_rmse, rf_test_r2]).transpose()
result.columns = ["Training RMSE", "Training R^2", "Testing RMSE", "Testing R^2"]

print(result)

joblib.dump(rf, "../models/model.pkl")


