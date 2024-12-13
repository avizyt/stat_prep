### **Step 2: Descriptive Statistics for Multivariate Data**

Let's dive deep into the **mathematical formulation** of descriptive statistics for multivariate data and implement each step with NumPy. We'll cover:

1. **Measures of Central Tendency (Multivariate Mean)**
2. **Measures of Dispersion (Variance and Covariance Matrix)**
3. **Correlation Matrix**
4. **Outlier Detection with Mahalanobis Distance**

---

### **1. Measures of Central Tendency: Multivariate Mean**

#### **Mathematical Formulation**:
For a dataset \(\mathbf{X} \in \mathbb{R}^{n \times p}\), where \(n\) is the number of observations and \(p\) is the number of variables:
- Each row of \(\mathbf{X}\) represents an observation: \(\mathbf{x}_i = [x_{i1}, x_{i2}, \ldots, x_{ip}]\).
- The multivariate mean is a vector of means for each variable:

\[
\boldsymbol{\mu} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i
\]

Where:
- \(\boldsymbol{\mu} = [\mu_1, \mu_2, \ldots, \mu_p]\), with \(\mu_j = \frac{1}{n} \sum_{i=1}^n x_{ij}\).

---

#### **Python Code (Multivariate Mean):**
```python
import numpy as np

# Generate synthetic data (100 observations, 3 variables)
np.random.seed(42)
data = np.random.rand(100, 3)  # 100 rows, 3 variables

# Calculate multivariate mean
mean_vector = np.mean(data, axis=0)
print("Multivariate Mean Vector:")
print(mean_vector)
```

---

### **2. Measures of Dispersion: Variance and Covariance Matrix**

#### **Variance**:
For a single variable \(x_j\), the variance is:

\[
\text{Var}(x_j) = \frac{1}{n} \sum_{i=1}^n (x_{ij} - \mu_j)^2
\]

#### **Covariance**:
For two variables \(x_j\) and \(x_k\), the covariance is:

\[
\text{Cov}(x_j, x_k) = \frac{1}{n-1} \sum_{i=1}^n (x_{ij} - \mu_j)(x_{ik} - \mu_k)
\]

#### **Covariance Matrix**:
The covariance matrix \(\Sigma\) is a symmetric \(p \times p\) matrix, where each entry \(\sigma_{jk}\) is the covariance between \(x_j\) and \(x_k\):

\[
\Sigma = 
\begin{bmatrix}
\text{Var}(x_1) & \text{Cov}(x_1, x_2) & \cdots & \text{Cov}(x_1, x_p) \\
\text{Cov}(x_2, x_1) & \text{Var}(x_2) & \cdots & \text{Cov}(x_2, x_p) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(x_p, x_1) & \text{Cov}(x_p, x_2) & \cdots & \text{Var}(x_p)
\end{bmatrix}
\]

---

#### **Python Code (Covariance Matrix):**
```python
# Calculate covariance matrix
cov_matrix = np.cov(data, rowvar=False)  # rowvar=False: columns are variables
print("Covariance Matrix:")
print(cov_matrix)
```

---

### **3. Correlation Matrix**

#### **Mathematical Formulation**:
The correlation coefficient between variables \(x_j\) and \(x_k\) is:

\[
\rho(x_j, x_k) = \frac{\text{Cov}(x_j, x_k)}{\sqrt{\text{Var}(x_j) \cdot \text{Var}(x_k)}}
\]

The correlation matrix is computed as:

\[
\mathbf{R} = 
\begin{bmatrix}
1 & \rho(x_1, x_2) & \cdots & \rho(x_1, x_p) \\
\rho(x_2, x_1) & 1 & \cdots & \rho(x_2, x_p) \\
\vdots & \vdots & \ddots & \vdots \\
\rho(x_p, x_1) & \rho(x_p, x_2) & \cdots & 1
\end{bmatrix}
\]

---

#### **Python Code (Correlation Matrix):**
```python
# Calculate correlation matrix
corr_matrix = np.corrcoef(data, rowvar=False)
print("Correlation Matrix:")
print(corr_matrix)
```

---

### **4. Outlier Detection with Mahalanobis Distance**

#### **Mathematical Formulation**:
Mahalanobis distance measures the distance of a point \(\mathbf{x}_i\) from the mean \(\boldsymbol{\mu}\), accounting for the covariance structure of the data:

\[
D_M^2(\mathbf{x}_i) = (\mathbf{x}_i - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}_i - \boldsymbol{\mu})
\]

Where:
- \(\Sigma^{-1}\): Inverse of the covariance matrix.
- Larger \(D_M^2\) values indicate potential outliers.

---

#### **Python Code (Mahalanobis Distance):**
```python
from scipy.spatial.distance import mahalanobis

# Calculate Mahalanobis distance for all points
inv_cov_matrix = np.linalg.inv(cov_matrix)
mahalanobis_distances = [
    mahalanobis(row, mean_vector, inv_cov_matrix) for row in data
]

# Identify potential outliers (threshold: 3 standard deviations)
threshold = 3
outliers = np.where(np.array(mahalanobis_distances) > threshold)[0]

print("Mahalanobis Distances:")
print(mahalanobis_distances)
print("\nOutlier Indices:")
print(outliers)
```

---

### **Summary Table of Functions**

| **Statistic**             | **Mathematical Formula**                                      | **NumPy Function**                  |
|----------------------------|-------------------------------------------------------------|-------------------------------------|
| Multivariate Mean          | \(\boldsymbol{\mu} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i\) | `np.mean(data, axis=0)`            |
| Covariance Matrix          | \(\text{Cov}(x_j, x_k) = \frac{1}{n-1} \sum (x_{ij} - \mu_j)(x_{ik} - \mu_k)\) | `np.cov(data, rowvar=False)`       |
| Correlation Matrix         | \(\rho(x_j, x_k) = \frac{\text{Cov}(x_j, x_k)}{\sqrt{\text{Var}(x_j) \cdot \text{Var}(x_k)}}\) | `np.corrcoef(data, rowvar=False)`  |
| Mahalanobis Distance       | \(D_M^2 = (\mathbf{x}_i - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}_i - \boldsymbol{\mu})\) | `scipy.spatial.distance.mahalanobis` |

---

Would you like to discuss any part of this in more detail, or should we move to a related concept such as **data preprocessing** or **multivariate normality**?