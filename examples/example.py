# %%
import numpy as np
import pandas as pd

from gnar.gnar import GNAR
# %%
# Generate some random data
ts = np.random.normal(0, 1, (100, 3))
A = np.array([[0, 1, 0], 
              [1, 0, 1],
              [0, 1, 0]])
# Fit a GNAR model
G = GNAR(A, p=2, s_vec=np.array([1, 1]), ts=ts, remove_mean=True, model_type="standard")
print(G)
# %%
# BIC and AIC
print(G.bic())
print(G.aic())
# %%
# Can fit the model to new data
ts = np.random.normal(0, 1, (100, 3))
G.fit(ts)
print(G)
# %%
# Draw the network
G.draw()
# %%
# Convert the GNAR model to a VAR model
V = G.to_var()
print(V)
# %%
# Simulate from the GNAR model
sim = G.simulate(100)
# %%
# Forecast using the GNAR model
preds = G.predict(ts[-3:].copy(), 5)
# %%
# Convert the data to a pandas dataframe and repeat the steps above
ts_df = pd.DataFrame(ts, columns=["A", "B", "C"])
G = GNAR(A, p=2, s_vec=np.array([1, 1]), ts=ts_df, model_type="standard")
print(G)
# %%
sim = G.predict(ts_df[-5:].copy(), 5)
# %%
# Initialise a GNAR model from some parameters rather than fitting the model. 
# First row is the mean, followed by the alpha and beta coefficients.
params = np.array([[0,0,0], [0.1, 0.3, -0.2], [0.4, -0.2, 0.1]])
G = GNAR(A, p=1, s_vec=np.array([1, 1]), parameters=params, sigma_2=0.5, model_type="local")
print(G)
# %%
