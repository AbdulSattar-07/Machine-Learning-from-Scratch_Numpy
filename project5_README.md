# Project 5: Machine Learning from Scratch - Step-by-Step Guide

## 🤖 Overview
This advanced project teaches machine learning implementation using only NumPy. You'll build neural networks, optimization algorithms, and ML models from scratch, combining all previous NumPy concepts.

## 🎯 Learning Objectives
- Master advanced linear algebra for ML
- Implement gradient computation and backpropagation
- Build neural networks from scratch
- Create optimization algorithms (SGD, Adam)
- Develop clustering and dimensionality reduction
- Practice memory-efficient array operations

## 📁 Project Structure
```
project5_ml_from_scratch/
├── project5_ml_from_scratch.py          # Main implementation
├── project5_formulas_details.md         # Mathematical formulas
├── project5_README.md                   # This guide
├── models/                              # Trained models
├── datasets/                            # Sample datasets
├── results/                             # Training results
└── ml_from_scratch_streamlit.py         # Interactive UI
```

## 🔧 Prerequisites
```bash
pip install numpy matplotlib plotly streamlit scikit-datasets
```

## 📊 Key NumPy Functions You'll Master

| Function | Purpose | ML Application |
|----------|---------|----------------|
| `np.dot()` | Matrix multiplication | Forward/backward propagation |
| `np.linalg.eig()` | Eigendecomposition | PCA, spectral methods |
| `np.random.multivariate_normal()` | Generate correlated data | Dataset creation |
| `np.argmax()` | Find maximum indices | Classification predictions |
| `np.clip()` | Bound values | Numerical stability |
| `np.concatenate()` | Join arrays | Batch processing |
| `np.newaxis` | Add dimensions | Broadcasting |
| `np.where()` | Conditional selection | Activation functions |

## 🚀 Step-by-Step Implementation

### Step 1: Foundation Classes
```python
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        # Clip for numerical stability
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

**NumPy Concepts:**
- Abstract base classes for code organization
- `np.maximum()` for element-wise maximum
- `np.clip()` for numerical stability
- Broadcasting with `keepdims=True`

### Step 2: Linear Regression Implementation
```python
class LinearRegression(BaseModel):
    """Linear Regression using gradient descent"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self._forward_pass(X)
            
            # Compute cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def _forward_pass(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, y, y_pred):
        n_samples = len(y)
        mse = np.mean((y_pred - y) ** 2)
        
        # Add regularization
        if self.regularization == 'l1':
            mse += self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            mse += self.lambda_reg * np.sum(self.weights ** 2)
        
        return mse
    
    def _compute_gradients(self, X, y, y_pred):
        n_samples = X.shape[0]
        
        # Basic gradients
        dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
        db = (2/n_samples) * np.sum(y_pred - y)
        
        # Add regularization gradients
        if self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += 2 * self.lambda_reg * self.weights
        
        return dw, db
    
    def predict(self, X):
        return self._forward_pass(X)
```

**NumPy Concepts:**
- `np.random.normal()` for weight initialization
- `np.dot()` for matrix multiplication
- `np.sign()` for L1 regularization
- Vectorized gradient computation

### Step 3: Logistic Regression Implementation
```python
class LogisticRegression(BaseModel):
    """Logistic Regression for binary classification"""
    
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
            z = np.dot(X, self.weights) + self.bias
            y_pred = ActivationFunctions.sigmoid(z)
            
            # Compute cost (cross-entropy)
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def _compute_cost(self, y, y_pred):
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        return cost
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return ActivationFunctions.sigmoid(z)
    
    def predict_classes(self, X):
        return (self.predict(X) >= 0.5).astype(int)
```

**NumPy Concepts:**
- Cross-entropy loss implementation
- `np.clip()` for numerical stability
- Boolean to integer conversion with `astype(int)`

### Step 4: Neural Network Implementation
```python
class NeuralNetwork(BaseModel):
    """Multi-layer perceptron neural network"""
    
    def __init__(self, layers, learning_rate=0.01, activation='relu', optimizer='sgd'):
        self.layers = layers  # List of layer sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.weights = []
        self.biases = []
        self.cost_history = []
        
        # Initialize optimizer
        if optimizer == 'adam':
            self.optimizer_obj = AdamOptimizer(learning_rate)
        else:
            self.optimizer_obj = SGDOptimizer(learning_rate)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights using Xavier/He initialization"""
        for i in range(len(self.layers) - 1):
            # He initialization for ReLU, Xavier for others
            if self.activation == 'relu':
                std = np.sqrt(2.0 / self.layers[i])
            else:
                std = np.sqrt(1.0 / self.layers[i])
            
            w = np.random.normal(0, std, (self.layers[i], self.layers[i+1]))
            b = np.zeros((1, self.layers[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _forward_propagation(self, X):
        """Forward pass through the network"""
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function
            if i == len(self.weights) - 1:  # Output layer
                if self.layers[-1] == 1:  # Binary classification
                    a = ActivationFunctions.sigmoid(z)
                else:  # Multi-class classification
                    a = ActivationFunctions.softmax(z)
            else:  # Hidden layers
                if self.activation == 'relu':
                    a = ActivationFunctions.relu(z)
                elif self.activation == 'tanh':
                    a = np.tanh(z)
                else:
                    a = ActivationFunctions.sigmoid(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def _backward_propagation(self, X, y, activations, z_values):
        """Backward pass to compute gradients"""
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
                    delta *= (1 - np.tanh(z_values[i-1])**2)
                else:
                    delta *= ActivationFunctions.sigmoid_derivative(z_values[i-1])
        
        return gradients_w, gradients_b
    
    def fit(self, X, y, epochs=1000, batch_size=32, validation_data=None):
        """Train the neural network"""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward propagation
                activations, z_values = self._forward_propagation(X_batch)
                
                # Compute cost
                cost = self._compute_cost(y_batch, activations[-1])
                epoch_cost += cost
                n_batches += 1
                
                # Backward propagation
                gradients_w, gradients_b = self._backward_propagation(X_batch, y_batch, activations, z_values)
                
                # Update parameters using optimizer
                self.optimizer_obj.update(self.weights + self.biases, gradients_w + gradients_b)
            
            # Record average cost
            avg_cost = epoch_cost / n_batches
            self.cost_history.append(avg_cost)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {avg_cost:.4f}")
                
                if validation_data:
                    X_val, y_val = validation_data
                    val_predictions = self.predict_classes(X_val)
                    val_accuracy = np.mean(val_predictions == y_val)
                    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    def _compute_cost(self, y, y_pred):
        """Compute cost function"""
        if self.layers[-1] == 1:  # Binary classification
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            cost = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        else:  # Multi-class classification
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            cost = -np.mean(np.sum(y * np.log(y_pred_clipped), axis=1))
        
        return cost
    
    def predict(self, X):
        activations, _ = self._forward_propagation(X)
        return activations[-1]
    
    def predict_classes(self, X):
        predictions = self.predict(X)
        if self.layers[-1] == 1:  # Binary classification
            return (predictions >= 0.5).astype(int).flatten()
        else:  # Multi-class classification
            return np.argmax(predictions, axis=1)
```

**NumPy Concepts:**
- Multi-dimensional array operations
- Advanced indexing and slicing
- Broadcasting for batch operations
- Matrix chain multiplication

This is the foundation - the complete implementation continues with optimizers, clustering, PCA, and evaluation metrics. Each component builds upon these NumPy fundamentals!