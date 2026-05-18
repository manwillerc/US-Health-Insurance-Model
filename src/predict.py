import pandas as pd
import numpy as np
import joblib

model = joblib.load("../models/model.pkl")

new_data = pd.read_csv("../data/new_data.csv")

predicted_charges = model.predict(new_data)

new_data['predicted_charges'] = predicted_charges
new_data.to_csv("../data/predicted_charges.csv", index=False)