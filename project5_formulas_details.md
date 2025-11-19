# Project 5: Machine Learning from Scratch - Formulas & Detailed Concepts

## 📐 Mathematical Formulas for Machine Learning

### 1. Linear Regression

#### Cost Function (Mean Squared Error)
```
J(θ) = (1/2m) Σᵢ₌₁ᵐ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```
Where:
- m = number of training examples
- hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
- θ = parameter vector

**Matrix Form:**
```
J(θ) = (1/2m)(Xθ - y)ᵀ(Xθ - y)
```

**NumPy Implementation:**
```python
def cost_function(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost
```

#### Gradient Descent
```
∂J/∂θⱼ = (1/m) Σᵢ₌₁ᵐ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
```

**Matrix Form:**
```
∇J(θ) = (1/m)Xᵀ(Xθ - y)
```

**Update Rule:**
```
θ := θ - α∇J(θ)
```

**NumPy Implementation:**
```python
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        predictions = X.dot(theta)
        gradient = (1/m) * X.T.dot(predictions - y)
        theta = theta - alpha * gradient
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history
```

#### Normal Equation (Analytical Solution)
```
θ = (XᵀX)⁻¹Xᵀy
```

**NumPy Implementation:**
```python
def normal_equation(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
```

### 2. Logistic Regression

#### Sigmoid Function
```
g(z) = 1 / (1 + e⁻ᶻ)
```
**Properties:**
- g(0) = 0.5
- g(∞) = 1
- g(-∞) = 0
- g'(z) = g(z)(1 - g(z))

**NumPy Implementation:**
```python
def sigmoid(z):
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

#### Hypothesis Function
```
hθ(x) = g(θᵀx) = 1 / (1 + e⁻θᵀˣ)
```

#### Cost Function (Cross-Entropy)
```
J(θ) = -(1/m) Σᵢ₌₁ᵐ [y⁽ⁱ⁾log(hθ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-hθ(x⁽ⁱ⁾))]
```

**NumPy Implementation:**
```python
def logistic_cost_function(X, y, theta):
    m = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    
    cost = -(1/m) * (y.dot(np.log(h)) + (1-y).dot(np.log(1-h)))
    return cost
```

#### Gradient
```
∂J/∂θⱼ = (1/m) Σᵢ₌₁ᵐ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
```

**NumPy Implementation:**
```python
def logistic_gradient(X, y, theta):
    m = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    gradient = (1/m) * X.T.dot(h - y)
    return gradient
```

### 3. Neural Networks

#### Forward Propagation
**Layer l:**
```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = g(z⁽ˡ⁾)
```

Where:
- W⁽ˡ⁾ = weight matrix for layer l
- b⁽ˡ⁾ = bias vector for layer l
- g = activation function
- a⁽⁰⁾ = X (input)

**NumPy Implementation:**
```python
def forward_propagation(X, weights, biases, activation_func):
    activations = [X]
    z_values = []
    
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = np.dot(activations[-1], W) + b
        z_values.append(z)
        
        if i == len(weights) - 1:  # Output layer
            a = sigmoid(z)  # or softmax for multi-class
        else:  # Hidden layers
            a = activation_func(z)
        
        activations.append(a)
    
    return activations, z_values
```

#### Backpropagation
**Output Layer Error:**
```
δ⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) ⊙ g'(z⁽ᴸ⁾)
```

**Hidden Layer Error:**
```
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ g'(z⁽ˡ⁾)
```

**Gradients:**
```
∂J/∂W⁽ˡ⁾ = (1/m)δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
∂J/∂b⁽ˡ⁾ = (1/m)Σδ⁽ˡ⁾
```

**NumPy Implementation:**
```python
def backward_propagation(X, y, activations, z_values, weights, activation_derivative):
    m = X.shape[0]
    gradients_W = []
    gradients_b = []
    
    # Output layer error
    delta = activations[-1] - y
    
    # Backpropagate through layers
    for i in reversed(range(len(weights))):
        # Gradients for current layer
        dW = (1/m) * np.dot(activations[i].T, delta)
        db = (1/m) * np.sum(delta, axis=0, keepdims=True)
        
        gradients_W.insert(0, dW)
        gradients_b.insert(0, db)
        
        # Error for previous layer
        if i > 0:
            delta = np.dot(delta, weights[i].T) * activation_derivative(z_values[i-1])
    
    return gradients_W, gradients_b
```

### 4. Activation Functions

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
ReLU'(x) = {1 if x > 0, 0 if x ≤ 0}
```

**NumPy Implementation:**
```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

#### Leaky ReLU
```
LeakyReLU(x) = {x if x > 0, αx if x ≤ 0}
```
Where α is a small positive constant (e.g., 0.01)

**NumPy Implementation:**
```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
```

#### Tanh (Hyperbolic Tangent)
```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
tanh'(x) = 1 - tanh²(x)
```

**NumPy Implementation:**
```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

#### Softmax (for Multi-class Classification)
```
softmax(xᵢ) = eˣⁱ / Σⱼ₌₁ᴷ eˣʲ
```

**NumPy Implementation:**
```python
def softmax(x):
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

### 5. Loss Functions

#### Mean Squared Error (MSE)
```
MSE = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)²
```

**Derivative:**
```
∂MSE/∂ŷ = (2/m)(ŷ - y)
```

#### Binary Cross-Entropy
```
BCE = -(1/m) Σᵢ₌₁ᵐ [yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
```

**Derivative:**
```
∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
```

#### Categorical Cross-Entropy
```
CCE = -(1/m) Σᵢ₌₁ᵐ Σⱼ₌₁ᴷ yᵢⱼlog(ŷᵢⱼ)
```

**NumPy Implementation:**
```python
def categorical_crossentropy(y_true, y_pred):
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

### 6. Regularization

#### L1 Regularization (Lasso)
```
J_reg = J + λ Σⱼ₌₁ⁿ |θⱼ|
```

**Gradient:**
```
∂J_reg/∂θⱼ = ∂J/∂θⱼ + λ × sign(θⱼ)
```

**NumPy Implementation:**
```python
def l1_regularization(weights, lambda_reg):
    return lambda_reg * np.sum(np.abs(weights))

def l1_gradient(weights, lambda_reg):
    return lambda_reg * np.sign(weights)
```

#### L2 Regularization (Ridge)
```
J_reg = J + λ Σⱼ₌₁ⁿ θⱼ²
```

**Gradient:**
```
∂J_reg/∂θⱼ = ∂J/∂θⱼ + 2λθⱼ
```

**NumPy Implementation:**
```python
def l2_regularization(weights, lambda_reg):
    return lambda_reg * np.sum(weights**2)

def l2_gradient(weights, lambda_reg):
    return 2 * lambda_reg * weights
```

#### Elastic Net (L1 + L2)
```
J_reg = J + λ₁ Σⱼ₌₁ⁿ |θⱼ| + λ₂ Σⱼ₌₁ⁿ θⱼ²
```

### 7. Optimization Algorithms

#### Stochastic Gradient Descent (SGD)
```
θ := θ - α∇J(θ)
```

#### SGD with Momentum
```
v := βv + (1-β)∇J(θ)
θ := θ - αv
```

**NumPy Implementation:**
```python
class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, gradients):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            self.velocity[i] = self.momentum * self.velocity[i] + (1 - self.momentum) * grad
            param -= self.lr * self.velocity[i]
```

#### AdaGrad
```
G := G + (∇J(θ))²
θ := θ - α/(√G + ε) × ∇J(θ)
```

#### RMSprop
```
E[g²] := βE[g²] + (1-β)(∇J(θ))²
θ := θ - α/(√E[g²] + ε) × ∇J(θ)
```

#### Adam (Adaptive Moment Estimation)
```
m := β₁m + (1-β₁)∇J(θ)
v := β₂v + (1-β₂)(∇J(θ))²
m̂ := m/(1-β₁ᵗ)
v̂ := v/(1-β₂ᵗ)
θ := θ - α × m̂/(√v̂ + ε)
```

**NumPy Implementation:**
```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, gradients):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

### 8. Batch Normalization

#### Forward Pass
```
μ = (1/m) Σᵢ₌₁ᵐ xᵢ
σ² = (1/m) Σᵢ₌₁ᵐ (xᵢ - μ)²
x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
yᵢ = γx̂ᵢ + β
```

**NumPy Implementation:**
```python
def batch_norm_forward(x, gamma, beta, epsilon=1e-8):
    # Calculate mean and variance
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    
    # Normalize
    x_norm = (x - mu) / np.sqrt(var + epsilon)
    
    # Scale and shift
    out = gamma * x_norm + beta
    
    # Cache for backward pass
    cache = (x, x_norm, mu, var, gamma, beta, epsilon)
    
    return out, cache
```

#### Backward Pass
```python
def batch_norm_backward(dout, cache):
    x, x_norm, mu, var, gamma, beta, epsilon = cache
    N = x.shape[0]
    
    # Gradients for gamma and beta
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # Gradient for normalized x
    dx_norm = dout * gamma
    
    # Gradient for variance
    dvar = np.sum(dx_norm * (x - mu) * -0.5 * (var + epsilon)**(-1.5), axis=0)
    
    # Gradient for mean
    dmu = np.sum(dx_norm * -1 / np.sqrt(var + epsilon), axis=0) + dvar * np.sum(-2 * (x - mu), axis=0) / N
    
    # Gradient for x
    dx = dx_norm / np.sqrt(var + epsilon) + dvar * 2 * (x - mu) / N + dmu / N
    
    return dx, dgamma, dbeta
```

### 9. Dropout

#### Forward Pass (Training)
```python
def dropout_forward(x, dropout_prob, training=True):
    if not training:
        return x, None
    
    # Create dropout mask
    mask = np.random.rand(*x.shape) > dropout_prob
    
    # Apply mask and scale
    out = x * mask / (1 - dropout_prob)
    
    return out, mask
```

#### Backward Pass
```python
def dropout_backward(dout, mask, dropout_prob):
    if mask is None:
        return dout
    
    return dout * mask / (1 - dropout_prob)
```

### 10. K-Means Clustering

#### Distance Calculation
```
d(x, c) = ||x - c||² = Σⱼ₌₁ⁿ (xⱼ - cⱼ)²
```

#### Centroid Update
```
cⱼ = (1/|Sⱼ|) Σ(xᵢ ∈ Sⱼ) xᵢ
```

**NumPy Implementation:**
```python
def kmeans_step(X, centroids):
    # Calculate distances to all centroids
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    
    # Assign points to closest centroid
    labels = np.argmin(distances, axis=0)
    
    # Update centroids
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(len(centroids))])
    
    return new_centroids, labels
```

### 11. Principal Component Analysis (PCA)

#### Covariance Matrix
```
C = (1/(m-1)) Σᵢ₌₁ᵐ (xᵢ - μ)(xᵢ - μ)ᵀ
```

#### Eigendecomposition
```
C = VΛVᵀ
```

Where:
- V = eigenvectors (principal components)
- Λ = eigenvalues (explained variance)

**NumPy Implementation:**
```python
def pca_transform(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    
    # Transform data
    X_transformed = np.dot(X_centered, components)
    
    # Explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_transformed, components, explained_variance_ratio
```

### 12. Model Evaluation Metrics

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### Precision
```
Precision = TP / (TP + FP)
```

#### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```

#### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### ROC-AUC
```python
def roc_auc_score(y_true, y_scores):
    # Sort by scores
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    
    # Calculate TPR and FPR
    tpr = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
    fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return auc
```

## 🧮 Advanced NumPy Techniques for ML

### 1. Vectorized Operations
```python
# Vectorized distance calculation
def euclidean_distance_vectorized(X1, X2):
    """
    X1: (n1, d) array
    X2: (n2, d) array
    Returns: (n1, n2) distance matrix
    """
    # Using broadcasting
    return np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2))

# Vectorized softmax with numerical stability
def stable_softmax(x):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### 2. Memory-Efficient Implementations
```python
# Generator for mini-batch processing
def mini_batch_generator(X, y, batch_size):
    """Generate mini-batches for training"""
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

# In-place operations to save memory
def inplace_normalize(X):
    """Normalize features in-place"""
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X
```

### 3. Numerical Stability Techniques
```python
# Log-sum-exp trick for numerical stability
def logsumexp(x, axis=None):
    """Numerically stable log-sum-exp"""
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

# Stable sigmoid implementation
def stable_sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
```

### 4. Advanced Broadcasting Patterns
```python
# Broadcasting for batch matrix operations
def batch_matrix_multiply(A, B):
    """
    A: (batch_size, n, k)
    B: (batch_size, k, m)
    Returns: (batch_size, n, m)
    """
    return np.matmul(A, B)

# Broadcasting for feature scaling
def feature_scale_broadcast(X, method='standardize'):
    """Scale features using broadcasting"""
    if method == 'standardize':
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif method == 'normalize':
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
```

Ye comprehensive mathematical foundation aapko machine learning algorithms ke saare concepts aur NumPy implementations ki deep understanding dega!