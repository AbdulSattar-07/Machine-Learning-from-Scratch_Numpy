# Project 5: Machine Learning from Scratch - Complete Guide

## 🤖 Project Overview
Build comprehensive machine learning algorithms from scratch using only NumPy. This advanced project combines all previous NumPy concepts to implement neural networks, optimization algorithms, and machine learning models without external ML libraries.

## 🎯 Learning Objectives
By completing this project, you will master:
- Advanced linear algebra operations for ML
- Gradient computation and backpropagation
- Matrix factorization techniques
- Broadcasting for batch operations
- Memory-efficient array operations
- Optimization algorithms (SGD, Adam, RMSprop)
- Neural network architectures
- Regularization and normalization techniques

## 📊 Mathematical Formulas Used

### 1. Linear Regression
```
ŷ = Xw + b
Cost Function: J(w,b) = (1/2m) Σ(ŷᵢ - yᵢ)²
Gradient: ∇w = (1/m) Xᵀ(ŷ - y)
         ∇b = (1/m) Σ(ŷᵢ - yᵢ)
```
**NumPy Implementation:**
```python
predictions = np.dot(X, weights) + bias
cost = np.mean((predictions - y)**2) / 2
dw = np.dot(X.T, (predictions - y)) / m
db = np.mean(predictions - y)
```

### 2. Logistic Regression
```
σ(z) = 1 / (1 + e^(-z))
ŷ = σ(Xw + b)
Cost: J = -(1/m) Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
```
**Sigmoid Function:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for stability
```

### 3. Neural Network Forward Propagation
```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = g(z⁽ˡ⁾)
```
Where g is the activation function (ReLU, sigmoid, tanh)

### 4. Backpropagation
```
δ⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) ⊙ g'(z⁽ᴸ⁾)
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ g'(z⁽ˡ⁾)
∇W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
∇b⁽ˡ⁾ = δ⁽ˡ⁾
```

### 5. Activation Functions
**ReLU:**
```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0, else 0
```

**Tanh:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

**Softmax:**
```
softmax(xᵢ) = e^xᵢ / Σⱼe^xⱼ
```

### 6. Optimization Algorithms

#### Gradient Descent
```
w := w - α∇w
```

#### Momentum
```
v := βv + (1-β)∇w
w := w - αv
```

#### Adam Optimizer
```
m := β₁m + (1-β₁)∇w
v := β₂v + (1-β₂)(∇w)²
m̂ := m/(1-β₁ᵗ)
v̂ := v/(1-β₂ᵗ)
w := w - α(m̂/(√v̂ + ε))
```

### 7. Regularization

#### L1 Regularization (Lasso)
```
J_reg = J + λΣ|wᵢ|
∇w_reg = ∇w + λ·sign(w)
```

#### L2 Regularization (Ridge)
```
J_reg = J + λΣwᵢ²
∇w_reg = ∇w + 2λw
```

#### Dropout
```
During training: aᵢ = aᵢ × mask / keep_prob
During inference: aᵢ = aᵢ × keep_prob
```

### 8. Batch Normalization
```
μ = (1/m)Σxᵢ
σ² = (1/m)Σ(xᵢ - μ)²
x̂ᵢ = (xᵢ - μ)/√(σ² + ε)
yᵢ = γx̂ᵢ + β
```

### 9. K-Means Clustering
```
Centroids Update: cⱼ = (1/|Sⱼ|)Σ(xᵢ ∈ Sⱼ) xᵢ
Assignment: argmin_j ||xᵢ - cⱼ||²
```

### 10. Principal Component Analysis (PCA)
```
Covariance Matrix: C = (1/m)XᵀX
Eigendecomposition: C = VΛVᵀ
Principal Components: PC = XV
```

## 🔄 Complete Project Steps

### Step 1: Linear Algebra Foundation
```python
class LinearAlgebra:
    @staticmethod
    def matrix_multiply(A, B):
        return np.dot(A, B)
    
    @staticmethod
    def matrix_inverse(A):
        return np.linalg.inv(A)
    
    @staticmethod
    def eigendecomposition(A):
        return np.linalg.eig(A)
    
    @staticmethod
    def svd(A):
        return np.linalg.svd(A)
```

### Step 2: Activation Functions
```python
class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

### Step 3: Loss Functions
```python
class LossFunctions:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / len(y_true)
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

### Step 4: Linear Regression Implementation
```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute cost
            cost = np.mean((y_pred - y)**2) / 2
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

### Step 5: Logistic Regression Implementation
```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            
            # Compute cost
            cost = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict_classes(self, X):
        return (self.predict(X) >= 0.5).astype(int)
```

### Step 6: Neural Network Implementation
```python
class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, activation='relu'):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = []
        self.biases = []
        self.cost_history = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # Xavier initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _forward_propagation(self, X):
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i == len(self.weights) - 1:  # Output layer
                if self.layers[-1] == 1:  # Binary classification
                    a = ActivationFunctions.sigmoid(z)
                else:  # Multi-class classification
                    a = ActivationFunctions.softmax(z)
            else:  # Hidden layers
                if self.activation == 'relu':
                    a = ActivationFunctions.relu(z)
                elif self.activation == 'tanh':
                    a = ActivationFunctions.tanh(z)
                else:
                    a = ActivationFunctions.sigmoid(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def _backward_propagation(self, X, y, activations, z_values):
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer error
        if self.layers[-1] == 1:  # Binary classification
            delta = activations[-1] - y.reshape(-1, 1)
        else:  # Multi-class classification
            delta = activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Gradients for current layer
            dw = np.dot(activations[i].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            
            # Error for previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                
                # Apply derivative of activation function
                if self.activation == 'relu':
                    delta *= ActivationFunctions.relu_derivative(z_values[i-1])
                elif self.activation == 'tanh':
                    delta *= ActivationFunctions.tanh_derivative(z_values[i-1])
                else:
                    delta *= ActivationFunctions.sigmoid_derivative(z_values[i-1])
        
        return gradients_w, gradients_b
    
    def fit(self, X, y, epochs=1000, batch_size=32):
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward propagation
                activations, z_values = self._forward_propagation(X_batch)
                
                # Compute cost
                if self.layers[-1] == 1:  # Binary classification
                    cost = LossFunctions.binary_crossentropy(y_batch.reshape(-1, 1), activations[-1])
                else:  # Multi-class classification
                    cost = LossFunctions.categorical_crossentropy(y_batch, activations[-1])
                
                epoch_cost += cost * len(X_batch)
                
                # Backward propagation
                gradients_w, gradients_b = self._backward_propagation(X_batch, y_batch, activations, z_values)
                
                # Update parameters
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * gradients_w[j]
                    self.biases[j] -= self.learning_rate * gradients_b[j]
            
            # Record average cost for epoch
            avg_cost = epoch_cost / n_samples
            self.cost_history.append(avg_cost)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {avg_cost:.4f}")
    
    def predict(self, X):
        activations, _ = self._forward_propagation(X)
        return activations[-1]
    
    def predict_classes(self, X):
        predictions = self.predict(X)
        if self.layers[-1] == 1:  # Binary classification
            return (predictions >= 0.5).astype(int)
        else:  # Multi-class classification
            return np.argmax(predictions, axis=1)
```

### Step 7: K-Means Clustering
```python
class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
    
    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign points to closest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        self.labels_ = labels
        return self
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
```

### Step 8: Principal Component Analysis
```python
class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

### Step 9: Optimization Algorithms
```python
class Optimizers:
    class SGD:
        def __init__(self, learning_rate=0.01, momentum=0.0):
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.velocity = None
        
        def update(self, params, gradients):
            if self.velocity is None:
                self.velocity = [np.zeros_like(p) for p in params]
            
            for i, (param, grad) in enumerate(zip(params, gradients)):
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
                param += self.velocity[i]
    
    class Adam:
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.learning_rate = learning_rate
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
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
                
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

### Step 10: Model Evaluation
```python
class ModelEvaluation:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred, average='binary'):
        if average == 'binary':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        else:
            # Multi-class precision
            classes = np.unique(y_true)
            precisions = []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            return np.mean(precisions)
    
    @staticmethod
    def recall(y_true, y_pred, average='binary'):
        if average == 'binary':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            # Multi-class recall
            classes = np.unique(y_true)
            recalls = []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))
                recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            return np.mean(recalls)
    
    @staticmethod
    def f1_score(y_true, y_pred, average='binary'):
        prec = ModelEvaluation.precision(y_true, y_pred, average)
        rec = ModelEvaluation.recall(y_true, y_pred, average)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        return cm
```

## 🧮 Advanced NumPy Concepts for ML

### 1. Broadcasting for Batch Operations
```python
# Efficient batch processing
def batch_operation(X, weights):
    # X: (batch_size, features)
    # weights: (features, output_size)
    return np.dot(X, weights)  # Broadcasting handles dimensions

# Batch normalization
def batch_normalize(X, epsilon=1e-8):
    mean = np.mean(X, axis=0, keepdims=True)
    var = np.var(X, axis=0, keepdims=True)
    return (X - mean) / np.sqrt(var + epsilon)
```

### 2. Memory-Efficient Operations
```python
# In-place operations to save memory
def inplace_relu(X):
    X[X < 0] = 0  # Modifies X directly
    return X

# Chunked processing for large datasets
def process_in_chunks(X, chunk_size=1000):
    n_samples = X.shape[0]
    results = []
    
    for i in range(0, n_samples, chunk_size):
        chunk = X[i:i+chunk_size]
        result = process_chunk(chunk)
        results.append(result)
    
    return np.concatenate(results, axis=0)
```

### 3. Advanced Indexing for ML
```python
# Fancy indexing for data sampling
def stratified_sample(X, y, n_samples_per_class):
    classes = np.unique(y)
    indices = []
    
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        sampled_indices = np.random.choice(cls_indices, n_samples_per_class, replace=False)
        indices.extend(sampled_indices)
    
    return X[indices], y[indices]

# Boolean indexing for data filtering
def filter_outliers(X, threshold=3):
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    return X[np.all(z_scores < threshold, axis=1)]
```

## 🔍 Practical Applications

### Supervised Learning
- Linear and logistic regression
- Multi-layer perceptrons
- Classification and regression tasks
- Feature selection and engineering

### Unsupervised Learning
- K-means clustering
- Principal Component Analysis
- Dimensionality reduction
- Anomaly detection

### Deep Learning Foundations
- Neural network architectures
- Backpropagation algorithm
- Optimization techniques
- Regularization methods

### Model Evaluation
- Cross-validation
- Performance metrics
- Hyperparameter tuning
- Model selection

## 💡 Key Learning Points

1. **Linear algebra is fundamental** to all machine learning algorithms
2. **Broadcasting enables efficient** batch processing
3. **Gradient computation** is the heart of optimization
4. **Matrix operations** provide computational efficiency
5. **Vectorization** is crucial for performance
6. **Memory management** matters for large datasets
7. **Numerical stability** prevents computational errors

## 🛠️ Tools and Libraries

### Core Libraries
- **NumPy** - All mathematical operations and array handling
- **Matplotlib** - Visualization of results and learning curves
- **SciPy** - Additional scientific computing functions

### Optional Enhancements
- **Numba** - Just-in-time compilation for speed
- **CuPy** - GPU acceleration
- **Joblib** - Parallel processing

## 📝 Project Structure
```
project5_ml_from_scratch/
├── project5_ml_from_scratch.py          # Main implementation
├── project5_formulas_details.md         # Mathematical formulas
├── project5_README.md                   # Step-by-step guide
├── models/                              # Trained models
├── datasets/                            # Sample datasets
├── results/                             # Training results
└── ml_from_scratch_streamlit.py         # Interactive UI
```

## 📊 Expected Outputs

### Model Performance
- Training and validation curves
- Accuracy, precision, recall metrics
- Confusion matrices
- ROC curves and AUC scores

### Visualizations
- Decision boundaries
- Clustering results
- Principal components
- Feature importance plots

### Analysis
- Convergence analysis
- Hyperparameter sensitivity
- Model comparison
- Performance benchmarks

This comprehensive guide provides everything needed to master machine learning implementation with NumPy from scratch through hands-on algorithm development!