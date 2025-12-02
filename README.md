# PPL-SI: Pretrained Penalized Lasso Selective Inference

**PPL-SI** is a Python package for conducting valid statistical inference in high-dimensional regression with transfer learning using Pretrained Lasso and DTransFusion algorithms. It implements selective inference methods to control the false positive rate (FPR) while maintaining high true positive rate (TPR) in feature selection after transfer learning.

## Features

- **Pretrained Lasso**: Transfer learning with explicit separation of shared and individual components
- **DTransFusion**: Debiased transfer fusion with source task aggregation and debiasing
- **Selective Inference**: Valid p-values controlling for model selection
- **Parallel Computing**: Efficient computation using joblib for large-scale problems

## Requirements & Installation

This package requires:
- **[NumPy](https://numpy.org/doc/stable/)** (`numpy`)
- **[mpmath](https://mpmath.org/)** (`mpmath`)
- **[skglm](https://contrib.scikit-learn.org/skglm/)** (`skglm`)
- **[SciPy](https://docs.scipy.org/doc/)** (`scipy`)
- **[joblib](https://joblib.readthedocs.io/)** (`joblib`)

Installation:
```bash
# Install from local directory
pip install -e .

# Or install dependencies manually
pip install numpy mpmath skglm scipy joblib
```

We recommend using Python 3.8 or later.

## Quick Start

### Pretrained Lasso

```python
from ppl_si import generate_synthetic_data, PPL_SI_randj

# Generate synthetic data
p = 300
K = 5
n_list = [100, 100, 100, 100, 200]

X_list, Y_list, true_betaK, Sigma_list = generate_synthetic_data(
    p=p, num_sh=10, num_inv=5, K=K, n_list=n_list,
    true_beta_sh=0.3, Gamma=0.1
)

# Compute p-value for a random selected feature
p_value = PPL_SI_randj(
    X_list=X_list,
    Y_list=Y_list,
    lambda_sh=95,
    lambda_K=15,
    rho=0.5,
    Sigma_list=Sigma_list
)

print(f"p-value: {p_value:.4f}")
```

### DTransFusion

```python
import numpy as np
from ppl_si import generate_synthetic_data, DTF_SI_randj

# Generate synthetic data
p = 300
K = 5
n_list = [100, 100, 100, 100, 200]

X_list, Y_list, true_betaK, Sigma_list = generate_synthetic_data(
    p=p, num_sh=10, num_inv=5, K=K, n_list=n_list,
    true_beta_sh=0.3, Gamma=0.1
)

# Set parameters
lambda_k_list = [np.sqrt(2 * np.log(p) / n_list[k]) for k in range(K)]
lambda_0 = np.sqrt(np.log(p) / sum(n_list))
lambda_tilde = 1.5 * np.sqrt(np.log(p) / n_list[-1])
qk_weights = [0.1 * np.sqrt(n_list[k] / sum(n_list)) for k in range(K - 1)]

# Compute p-value
p_value = DTF_SI_randj(
    X_list=X_list,
    Y_list=Y_list,
    lambda_k_list=lambda_k_list,
    lambda_0=lambda_0,
    lambda_tilde=lambda_tilde,
    qk_weights=qk_weights,
    Sigma_list=Sigma_list
)

print(f"p-value: {p_value:.4f}")
```

## Example Notebooks

We provide several Jupyter notebooks demonstrating package usage in the `examples/` directory:

- `ex1_p_value_PPL.ipynb` - Computing p-values with Pretrained Lasso
- `ex2_p_value_DTF.ipynb` - Computing p-values with DTransFusion
- `ex3_quick_start.ipynb` - Quick start guide for both methods

## Package Structure

```
ppl_si/
├── gen_data.py          # Data generation functions
├── algorithms.py        # Core algorithms (PretrainedLasso, DTransFusion)
├── utils.py             # Utility functions
├── sub_prob.py          # Subproblem solvers
├── PPL_SI.py           # Main selective inference functions
└── __init__.py         # Package initialization
```

## Main Functions

### Selective Inference

- `PPL_SI(X_list, Y_list, lambda_sh, lambda_K, rho, Sigma_list, ...)` - Compute p-values for all selected features using Pretrained Lasso
- `PPL_SI_randj(...)` - Compute p-value for a random selected feature
- `DTF_SI(X_list, Y_list, lambda_k_list, lambda_0, lambda_tilde, qk_weights, Sigma_list, ...)` - Compute p-values using DTransFusion
- `DTF_SI_randj(...)` - Compute p-value for a random selected feature

### Core Algorithms

- `PretrainedLasso(X, Y, XK, YK, lambda_sh, lambda_K, rho, n, nK)` - Pretrained Lasso estimator
- `DTransFusion(X_tilde, Y_tilde, XK, YK, q_tilde, lambda_tilde, P)` - DTransFusion estimator
- `source_estimator(Xk, Yk, lambda_tilde_k)` - Debiased source task estimator

### Data Generation

- `generate_synthetic_data(p, num_sh, num_inv, K, n_list, ...)` - Generate synthetic data for transfer learning

## Methodology

### Pretrained Lasso

The Pretrained Lasso algorithm separates the estimation into two steps:

1. **Shared component**: Estimate β_shared from pooled source data
2. **Individual component**: Estimate β_individual from target residuals with adaptive weights

The final estimator combines both: β_target = (1-ρ)·β_shared + β_individual

### DTransFusion

DTransFusion uses a two-stage approach:

1. **Source estimation**: Debias source task estimates using inverse covariance
2. **Co-training**: Aggregate debiased source estimates with weighted Lasso
3. **Debiasing**: Remove bias in target task estimation

The final estimator: β_target = w_hat + δ_hat

### Selective Inference

Both methods use:
- **Polyhedral constraints**: Characterize the selection event
- **Divide-and-conquer**: Efficient parallel computation of selection intervals
- **Truncated normal**: Compute valid p-values conditional on selection

## Citation

If you use this package, please cite the related papers on transfer learning and selective inference.

## License

This package is for research and educational purposes.

## Contact

For questions or issues, please open an issue on the repository.
