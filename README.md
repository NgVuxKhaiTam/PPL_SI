# PPL-SI: Pretrained Penalized Lasso Selective Inference

**PPL-SI** is a Python package for conducting valid statistical inference in high-dimensional regression with transfer/distributed learning using Pretrained Lasso and DTransFusion algorithms. It implements selective inference methods to control the false positive rate (FPR) while maintaining high true positive rate (TPR) in feature selection after transfer learning.

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

### Selective Inference

Both methods use:
- **Polyhedral constraints**: Characterize the selection event
- **Divide-and-conquer**: Efficient parallel computation of selection intervals
- **Truncated normal**: Compute valid p-values conditional on selection
