# Load libriaries and functions.
import pandas as pd
import numpy as np
import tensorflow as tf
tfk = tf.keras
tf.keras.backend.set_floatx("float64")

import tensorflow_probability as tfp
tfd = tfp.distributions

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest# Define helper functions.
scaler = StandardScaler()

detector = IsolationForest(n_estimators=1000, behaviour="deprecated", contamination="auto", random_state=0)
neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

#%%
# Load data and keep only first six months due to drift.
data = pd.read_excel("data/AirQualityUCI.xlsx")
data = data[data["Date"] <= "2004-09-10"]

#%%
# Select columns and remove rows with missing values.
columns = ["PT08.S1(CO)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "AH", "CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
data = data[columns].dropna(axis=0)             # Scale data to zero mean and unit variance.
X_t = scaler.fit_transform(data)                # Remove outliers.
is_inlier = detector.fit_predict(X_t)
X_t = X_t[(is_inlier > 0),:]                    # Restore frame.
dataset = pd.DataFrame(X_t, columns=columns)    # Select labels for inputs and outputs.
inputs = ["PT08.S1(CO)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "AH"]
outputs = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]