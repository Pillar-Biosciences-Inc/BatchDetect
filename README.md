# BatchDetect
Detecting batch heterogeneity via likelihoods

BatchDetect addresses the challenge of detecting batch effects and heterogeneity in datasets, particularly when sample sizes are small. Traditional asymptotic tests often fail in these regimes, leading to unreliable conclusions. The core philosophy of this package is to evaluate the distribution of sample likelihoods (or scores) to determine if they arise from a single homogeneous process or if there is evidence of multiple underlying clusters representing distinct batches.

The methodology relies on fitting "Heavy Mixture" models—supporting robust distributions like Laplace and Student-t—to the data. We explicitly test the hypothesis of a single component ($K=1$) versus two components ($K=2$). To ensure statistical rigor even with limited data, we employ a parametric bootstrap likelihood ratio test (`parametric_bootstrap_lrt`), which constructs a null distribution by resampling. Additionally, the package includes methods for clustering on correlation matrices (using hierarchical, spectral, or Leiden algorithms) to identify potential batch structures before formal testing.

## Repository Structure

- `batchdetect/`: Contains the main source code for the package, including mixture models, model selection, and clustering algorithms.
- `tests/`: Contains the comprehensive unit test suite to ensure meaningful coverage and correctness.
- `paper/`: Includes all specific code, notebooks, and experiments associated with the accompanying publication for this repository.

## Installation

### 🔧 Requirements

- Python ≥ 3.8
- NumPy, SciPy, pandas, matplotlib
- `pytest` (for testing)
- particles

You can install everything using pip:

```bash
# Clone the repository
git clone https://github.com/Pillar-Biosciences-Inc/BatchDetect.git
cd BatchDetect

# Install dependencies and the package
conda create -n batchdetect python=3.11
conda activate batchdetect
pip install -e .
```

## Usage

### Mixture Modeling
Fit a Heavy Mixture Model (e.g., Gaussian, Laplace, Student-t) to 1D data.

```python
import numpy as np
from batchdetect.mixture import HeavyMixture

# Generate synthetic data
X = np.concatenate([
    np.random.normal(0, 1, (100, 1)),
    np.random.normal(5, 1, (100, 1))
])

# Fit model
model = HeavyMixture(n_components=2, component_distribution='gaussian', random_state=42)
model.fit(X)

print(f"Weights: {model.weights_}")
print(f"Means: {model.means_.ravel()}")
```

### Sample Correlation
Compute Pearson and Spearman correlations between samples (rows).

```python
import numpy as np
from batchdetect.sample_correlation import get_correlations

X = np.random.randn(10, 5) # 10 samples, 5 features
pearson, spearman = get_correlations(X)
```

### Clustering
Cluster samples based on their correlation matrix (requires `igraph` and `leidenalg`).

```python
import numpy as np
from batchdetect.clustering import cluster_hierarchical_corr

# Example correlation matrix
corr = np.array([
    [1.0, 0.9, 0.1],
    [0.9, 1.0, 0.1],
    [0.1, 0.1, 1.0]
])

labels = cluster_hierarchical_corr(corr, n_clusters=2)
print(f"Cluster labels: {labels}")
```

### P-value Evaluation
Evaluate if p-values from a statistical test are conservative compared to a uniform distribution.

```python
from batchdetect.pvalue_evaluation import conservativeness_bound

# Example p-values from a null distribution
null_pvals = [0.01, 0.04, 0.2, 0.5]

result = conservativeness_bound(null_pvals, alpha=0.05)
print(f"Is conservative: {result['is_conservative']}")
# result contains bounds and estimates
```

