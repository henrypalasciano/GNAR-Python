# %%
import numpy as np
import pandas as pd

from gnar.gnar import GNAR
# %%
# Adjacency matrix of the five-net
A = np.array([[0, 0, 0, 1, 1], 
              [0, 0, 1, 1, 0],
              [0, 1, 0, 1, 0],
              [1, 1, 1, 0, 0],
              [1, 0, 0, 0, 0]])
# Parameters of the GNAR model - first row is the mean, followed by the alpha and beta coefficients. The columns are all equal since the GNAR model is global.
coeffs = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], 
                   [0.2, 0.2, 0.2, 0.2, 0.2],
                   [0.5, 0.5, 0.5, 0.5, 0.5],
                   [-0.1, -0.1, -0.1, -0.1, -0.1]])
# Convert the parameters to a pandas dataframe, so that the GNAR model stores the time series names
coeffs_df = pd.DataFrame(coeffs, columns=["A", "B", "C", "D", "E"]) 
five_net = GNAR(A=A, p=2, s=np.array([1, 1]), coeffs=coeffs_df, mean=0, sigma_2=1, model_type="global")
print(five_net)
# %%
# Draw the network
five_net.draw()
# %%
# Simulate from the five-net GNAR model
five_net_ts = five_net.simulate(1000)
# Fit a new GNAR model to the simulated data
five_net_ts_df = pd.DataFrame(five_net_ts, columns=["A", "B", "C", "D", "E"])
five_net_fit = GNAR(A, p=2, s=np.array([1, 1]), ts=five_net_ts_df, model_type="global")
print(five_net_fit)
# %%
# Compute tha AIC and BIC of the fit
print("AIC: ", np.round(five_net_fit.aic(), 3))
print("BIC: ", np.round(five_net_fit.bic(), 3))
# %%
# Simulate from the true model and forecast the next 5 time points
sim = pd.DataFrame(five_net.simulate(10), columns=["A", "B", "C", "D", "E"])
preds = five_net.predict(sim, 5)
# Since the input data is a pandas dataframe, the output is also a pandas dataframe with multi-index columns where the first level is the node and the second level is the horizon.
preds
# %%
# Convert the five-net GNAR model to a VAR model
five_net_var = five_net.to_var()
# Print the VAR model, the coefficents of each lag are grouped together
print(five_net_var)
# %%
