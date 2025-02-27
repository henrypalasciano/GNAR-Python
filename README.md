# GNAR Python: A Python implementation of Generalised Network Autoregressive Processes

This repository provides a **Python implementation** of the **Generalised Network Autoregressive (GNAR) model**,  
as described in the paper:  
📄 **[Generalized Network Autoregressive Processes and the GNAR Package](https://doi.org/10.18637/jss.v096.i05)**  

GNAR processes are a class of autoregressive models that describe the behavior of **multivariate time series on graphs**. Each univariate time series represents a **node** on the graph, with information flowing between them via the **edges**. The graph imposes additional constraints on the parameters of the GNAR process, depending on the model class.  

- In global - $\alpha$ models, all parameters are shared between nodes.  
- In standard (or local - $\alpha$) models, only the $\beta$ (neighbour set) coefficients are shared whereas the $\alpha$ ( autoregressive coefficients) is node specific.  
- In local - $\alpha\beta$ models, all parameters are node specific.  

**Note:** The current implementation **only supports unweighted networks**.  
**Support for weighted graphs will be added in a future update.**

---

## Installation  

To install GNAR-Python, clone this repository and install it using `pip`:

```bash
git clone https://github.com/henrypalasciano/GNAR-Python.git
cd GNAR-Python
pip install .
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/henrypalasciano/GNAR-Python.git
```

---

## 📖 Example Usage  

Below is a simple example of how to use the **GNAR-Python** package to fit a Generalised Network Autoregressive (GNAR) model, generate forecasts, and simulate data.

```python
import numpy as np
from gnar.gnar import GNAR

# Generate synthetic time series data (100 time steps, 3 nodes)
ts = np.random.normal(0, 1, (100, 3))

# Define an adjacency matrix for the network
A = np.array([[0, 1, 0], 
              [1, 0, 1],
              [0, 1, 0]])

# Fit a standard GNAR(2,[1,1]) process to the time series data
G = GNAR(A, p=2, s=np.array([1, 1]), ts=ts, demean=True, model_type="standard")
print(G)

# Compute model selection criteria
print("BIC:", G.bic())
print("AIC:", G.aic())

# Simulate 100 time steps from the fitted GNAR model
simulated_data = G.simulate(100)

# Forecast the next 5 time steps from some time series data (here h is the forecast horizon)
ts = np.random.normal(0, 1, (10, 3))
predictions = G.predict(ts=ts, h=5)

# Alternatively forecast directly from the last observation of the multivariate time series the model was fit to
predictions = G.predict(h=5)

# Visualise the graph
G.draw()
```

---

## 📂 Repository Structure  

The repository is organised as follows:

```plaintext
📂 GNAR-Python/
 ┣ 📂 gnar/            # Core GNAR model implementation
 ┣ 📂 examples/        # Example scripts demonstrating GNAR usage
 ┣ 📂 tests/           # Unit tests for model validation
 ┣ 📜 README.md        # Project documentation
 ┣ 📜 requirements.txt # List of dependencies
 ┣ 📜 setup.py         # Installation setup file
```

---

## Citing  

If you use **GNAR-Python** in your research, please cite the following paper:  

```bibtex
@article{GNAR,
  title={{Generalized Network Autoregressive Processes} and the {GNAR} Package},
  volume={96},
  url={https://www.jstatsoft.org/index.php/jss/article/view/v096i05},
  doi={10.18637/jss.v096.i05},
  number={5},
  journal={Journal of Statistical Software},
  author={Knight, M. and Leeming, K. and Nason, G. P. and Nunes, M.},
  year={2020},
  pages={1–36}
}
```

and Python implementation:

```bibtex
@software{GNAR-Python,
  author = {Palasciano, H. A.},
  title = {{GNAR Python}: A Python implementation of Generalised Network Autoregressive Processes},
  year = {2024},
  url = {https://github.com/henrypalasciano/GNAR-Python},
  version = {1.0}
}
```

---

## Contact  

**Henry Antonio Palasciano**  
📧 Email: [henry.palasciano17@imperial.ac.uk](mailto:henry.palasciano17@imperial.ac.uk)

