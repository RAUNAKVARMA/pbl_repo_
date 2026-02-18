import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib

df = pd.read_csv("../data/hybrid_wpt_dataset.csv")

X = df[["distance", "alignment", "soc", "grid_load"]]
y = df["mode"]

log_model = joblib.load("../models/logistic_model.pkl")

y_prob = log_model.predict_proba(X)[:,1]

fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("../figures/roc_curve.png")

print("Evaluation complete.")
