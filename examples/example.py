# %%
import numpy as np
import pandas as pd

from gnar.gnar import GNAR
from gnar.var import VAR
# %%
# Generate some random data
ts = np.random.normal(0, 1, (100, 3))
# Create an adjacency matrix
A = np.array([[0, 1, 0], 
              [1, 0, 1],
              [0, 1, 0]])
# Fit a GNAR model
G = GNAR(A, p=2, s=np.array([1, 1]), ts=ts, demean=True, method="OLS", model_type="standard")
print(G)
# %%
# BIC and AIC
print(G.bic())
print(G.aic())
# %%
# We can also re-fit the model to new data
ts = np.random.normal(0, 1, (100, 3))
G.fit(ts, demean=True, method="OLS")
print(G)
# %%
# Draw the network
G.draw()
# %%
# Convert the GNAR model to a VAR model
V = G.to_var()
print(V)
# %%
# Simulate from the GNAR model - the output is a numpy array of shape 
# (n, d) where n is the number of observations and d the number of nodes
sim = G.simulate(100)
sim
# %%
# Forecast using the GNAR model - since the input is a numpy array, 
# the output is a numpy array of shape (n - p + 1, d, h)
preds = G.predict(ts=np.random.normal(0, 1, (10, 3)), h=5)
preds
# %%
# Can also fit the model using Yule-Walker equations
G2 = GNAR(A, p=2, s=np.array([1, 1]), ts=ts, demean=True, method="YW", model_type="standard")
# %%
