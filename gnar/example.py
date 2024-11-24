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
G = GNAR(A, p=2, s_vec=np.array([2, 1]), ts=ts)
# %%
print(G)
# %%
G_sim = G.simulate(100)
# %%
# Forecast using the GNAR model
G.predict(ts[-3:].copy(), 5)
# %%
# Convert the data to a pandas dataframe and repeat the steps above
ts_df = pd.DataFrame(ts, columns=["A", "B", "C"])
# %%
G = GNAR(A, p=2, s_vec=np.array([1, 1]), ts=ts_df)
# %%
print(G)
# %%
G.predict(ts_df[-5:].copy(), 5)
# %%
G = GNAR(A, p=2, s_vec=np.array([1, 1]), parameters=np.array([[0,0,0], [1,2,3],[2,3,4],[4,5,6],[7,8,9]]))
# %%
print(G)
# %%
G = GNAR(A, p=2, s_vec=np.array([1, 1]), parameters=np.array([[0,0,0], [1,2,3],[-2,3,4],[4,5,-6],[7,-8,9]]) / 10, sigma_2=0.5)
# %%
