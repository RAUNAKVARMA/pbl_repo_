import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("../data/hybrid_wpt_dataset.csv")

X = df[["distance", "alignment", "soc", "grid_load"]]
y = df["mode"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(64,32,16),
                          max_iter=500)
mlp_model.fit(X_train, y_train)

joblib.dump(log_model, "../models/logistic_model.pkl")
joblib.dump(mlp_model, "../models/mlp_model.pkl")

print("Models trained and saved.")
