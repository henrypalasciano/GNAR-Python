# %%
import numpy as np
import pandas as pd

from gnar import GNAR
# %%
# Generate some random data
ts = np.random.normal(0, 1, (100, 3))
A = np.array([[0, 1, 0], 
              [1, 0, 1],
              [0, 1, 0]])
# %%
# Fit the GNAR model
G = GNAR(ts, A, p=2, s_vec=np.array([1, 1]))
# %%
G.mu
# %%
G.coefficients
# %%
# Forecast using the GNAR model
G.predict(ts[-3:].copy(), 5)
# %%
# Convert the data to a pandas dataframe and repeat the steps above
ts_df = pd.DataFrame(ts, columns=["A", "B", "C"])
# %%
G = GNAR(ts_df, A, p=2, s_vec=np.array([1, 1]))
# %%
G.coefficients
# %%
G.predict(ts_df[-5:].copy(), 5)
# %%
