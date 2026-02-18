import numpy as np
import pandas as pd

np.random.seed(42)

N = 20000

d = np.random.uniform(0.05, 2.0, N)
alignment = np.random.uniform(0, 0.2, N)
soc = np.random.uniform(0.1, 1.0, N)
grid_load = np.random.uniform(0.2, 1.0, N)

# Efficiency models
eta_ind = np.exp(-3*d) * np.exp(-5*alignment)
eta_mw = 0.75 * np.exp(-0.2*d)

mode = (eta_mw > eta_ind).astype(int)

df = pd.DataFrame({
    "distance": d,
    "alignment": alignment,
    "soc": soc,
    "grid_load": grid_load,
    "eta_ind": eta_ind,
    "eta_mw": eta_mw,
    "mode": mode
})

df.to_csv("../data/hybrid_wpt_dataset.csv", index=False)

print("Dataset generated.")
