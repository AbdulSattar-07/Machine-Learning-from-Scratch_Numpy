"""
Project 5: Machine Learning from Scratch with NumPy
Complete ML algorithms implementation using only NumPy
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class MLFromScratch:
    """Complete Machine Learning Framework using NumPy"""
    
    def __init__(self):
        self.models = {}
        self.datasets = {}
        self.create_directories()
        print("🤖 ML from Scratch Framework initialized")
    
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs('models', exist_ok=True)
        os.makedirs('datasets', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        print("📁 Created directories: models/, datasets/, results/")
    
    # ==================== BASE CLASSES ====================
    
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
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                predictions = np.argmax(predictions, axis=1)
            return np.mean(predictions == y)
    
    # ==================== ACTIVATION FUNCTIONS ====================
    
    class ActivationFunctions:
        """Collection of activation functions and derivatives"""
        
        @staticmethod
        def relu(x):
            return np.maximum(0, x)
        
        @staticmethod
        def relu_derivative(x):
            return (x > 0).astype(float)
        
        @staticmethod
        def sigmoid(x):
            x_clipped = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x_clipped))
        
        @staticmethod
        def sigmoid_derivative(x):
            s = MLFromScratch.ActivationFunctions.sigmoid(x)
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
    
    # ==================== LOSS FUNCTIONS ====================
    
    class LossFunctions:
        """Collection of loss functions"""
        
        @staticmethod
        def mean_squared_error(y_true, y_pred):
            return np.mean((y_true - y_pred)**2)
        
        @staticmethod
        def binary_crossentropy(y_true, y_pred):
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        
        @staticmethod
        def categorical_crossentropy(y_true, y_pred):
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))  
  
    # ==================== OPTIMIZERS ====================
    
    class SGDOptimizer:
        """Stochastic Gradient Descent Optimizer"""
        
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
    
    class AdamOptimizer:
        """Adam Optimizer"""
        
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
    
    # ==================== LINEAR REGRESSION ====================
    
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
            n_samples, n_features = X.shape
            self.weights = np.random.normal(0, 0.01, n_features)
            self.bias = 0
            
            for i in range(self.n_iterations):
                y_pred = self._forward_pass(X)
                cost = self._compute_cost(y, y_pred)
                self.cost_history.append(cost)
                
                dw, db = self._compute_gradients(X, y, y_pred)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        def _forward_pass(self, X):
            return np.dot(X, self.weights) + self.bias
        
        def _compute_cost(self, y, y_pred):
            n_samples = len(y)
            mse = np.mean((y_pred - y) ** 2)
            
            if self.regularization == 'l1':
                mse += self.lambda_reg * np.sum(np.abs(self.weights))
            elif self.regularization == 'l2':
                mse += self.lambda_reg * np.sum(self.weights ** 2)
            
            return mse
        
        def _compute_gradients(self, X, y, y_pred):
            n_samples = X.shape[0]
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            if self.regularization == 'l1':
                dw += self.lambda_reg * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += 2 * self.lambda_reg * self.weights
            
            return dw, db
        
        def predict(self, X):
            return self._forward_pass(X)
    
    # ==================== LOGISTIC REGRESSION ====================
    
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
            self.weights = np.zeros(n_features)
            self.bias = 0
            
            for i in range(self.n_iterations):
                z = np.dot(X, self.weights) + self.bias
                y_pred = MLFromScratch.ActivationFunctions.sigmoid(z)
                
                cost = self._compute_cost(y, y_pred)
                self.cost_history.append(cost)
                
                dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
                db = (1/n_samples) * np.sum(y_pred - y)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        def _compute_cost(self, y, y_pred):
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            cost = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            return cost
        
        def predict(self, X):
            z = np.dot(X, self.weights) + self.bias
            return MLFromScratch.ActivationFunctions.sigmoid(z)
        
        def predict_classes(self, X):
            return (self.predict(X) >= 0.5).astype(int) 
   
    # ==================== NEURAL NETWORK ====================
    
    class NeuralNetwork(BaseModel):
        """Multi-layer perceptron neural network"""
        
        def __init__(self, layers, learning_rate=0.01, activation='relu', optimizer='sgd'):
            self.layers = layers
            self.learning_rate = learning_rate
            self.activation = activation
            self.optimizer = optimizer
            self.weights = []
            self.biases = []
            self.cost_history = []
            
            if optimizer == 'adam':
                self.optimizer_obj = MLFromScratch.AdamOptimizer(learning_rate)
            else:
                self.optimizer_obj = MLFromScratch.SGDOptimizer(learning_rate)
            
            self._initialize_parameters()
        
        def _initialize_parameters(self):
            for i in range(len(self.layers) - 1):
                if self.activation == 'relu':
                    std = np.sqrt(2.0 / self.layers[i])
                else:
                    std = np.sqrt(1.0 / self.layers[i])
                
                w = np.random.normal(0, std, (self.layers[i], self.layers[i+1]))
                b = np.zeros((1, self.layers[i+1]))
                
                self.weights.append(w)
                self.biases.append(b)
        
        def _forward_propagation(self, X):
            activations = [X]
            z_values = []
            
            for i in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                z_values.append(z)
                
                if i == len(self.weights) - 1:  # Output layer
                    if self.layers[-1] == 1:
                        a = MLFromScratch.ActivationFunctions.sigmoid(z)
                    else:
                        a = MLFromScratch.ActivationFunctions.softmax(z)
                else:  # Hidden layers
                    if self.activation == 'relu':
                        a = MLFromScratch.ActivationFunctions.relu(z)
                    elif self.activation == 'tanh':
                        a = MLFromScratch.ActivationFunctions.tanh(z)
                    else:
                        a = MLFromScratch.ActivationFunctions.sigmoid(z)
                
                activations.append(a)
            
            return activations, z_values
        
        def _backward_propagation(self, X, y, activations, z_values):
            m = X.shape[0]
            gradients_w = []
            gradients_b = []
            
            if self.layers[-1] == 1:
                delta = activations[-1] - y.reshape(-1, 1)
            else:
                delta = activations[-1] - y
            
            for i in reversed(range(len(self.weights))):
                dw = np.dot(activations[i].T, delta) / m
                db = np.mean(delta, axis=0, keepdims=True)
                
                gradients_w.insert(0, dw)
                gradients_b.insert(0, db)
                
                if i > 0:
                    delta = np.dot(delta, self.weights[i].T)
                    
                    if self.activation == 'relu':
                        delta *= MLFromScratch.ActivationFunctions.relu_derivative(z_values[i-1])
                    elif self.activation == 'tanh':
                        delta *= MLFromScratch.ActivationFunctions.tanh_derivative(z_values[i-1])
                    else:
                        delta *= MLFromScratch.ActivationFunctions.sigmoid_derivative(z_values[i-1])
            
            return gradients_w, gradients_b
        
        def fit(self, X, y, epochs=1000, batch_size=32):
            n_samples = X.shape[0]
            
            for epoch in range(epochs):
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                epoch_cost = 0
                n_batches = 0
                
                for i in range(0, n_samples, batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    
                    activations, z_values = self._forward_propagation(X_batch)
                    cost = self._compute_cost(y_batch, activations[-1])
                    epoch_cost += cost
                    n_batches += 1
                    
                    gradients_w, gradients_b = self._backward_propagation(X_batch, y_batch, activations, z_values)
                    
                    all_params = self.weights + self.biases
                    all_gradients = gradients_w + gradients_b
                    self.optimizer_obj.update(all_params, all_gradients)
                
                avg_cost = epoch_cost / n_batches
                self.cost_history.append(avg_cost)
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Cost: {avg_cost:.4f}")
        
        def _compute_cost(self, y, y_pred):
            if self.layers[-1] == 1:
                return MLFromScratch.LossFunctions.binary_crossentropy(y.reshape(-1, 1), y_pred)
            else:
                return MLFromScratch.LossFunctions.categorical_crossentropy(y, y_pred)
        
        def predict(self, X):
            activations, _ = self._forward_propagation(X)
            return activations[-1]
        
        def predict_classes(self, X):
            predictions = self.predict(X)
            if self.layers[-1] == 1:
                return (predictions >= 0.5).astype(int).flatten()
            else:
                return np.argmax(predictions, axis=1)
        
        def predict_proba(self, X):
            """Return prediction probabilities"""
            return self.predict(X)    

    # ==================== K-MEANS CLUSTERING ====================
    
    class KMeans:
        """K-Means clustering algorithm"""
        
        def __init__(self, k=3, max_iters=100, random_state=None):
            self.k = k
            self.max_iters = max_iters
            self.random_state = random_state
            self.centroids = None
            self.labels_ = None
            self.inertia_ = None
            self.n_iterations = 0
        
        def fit(self, X):
            if self.random_state:
                np.random.seed(self.random_state)
            
            n_samples, n_features = X.shape
            
            # Initialize centroids randomly
            self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
            
            for iteration in range(self.max_iters):
                # Assign points to closest centroid
                distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
                labels = np.argmin(distances, axis=0)
                
                # Update centroids
                new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
                
                # Check for convergence
                if np.allclose(self.centroids, new_centroids):
                    break
                
                self.centroids = new_centroids
                self.n_iterations = iteration + 1
            
            self.labels_ = labels
            
            # Calculate inertia (within-cluster sum of squares)
            self.inertia_ = 0
            for i in range(self.k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    self.inertia_ += np.sum((cluster_points - self.centroids[i])**2)
            
            return self
        
        def predict(self, X):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            return np.argmin(distances, axis=0)
        
        def fit_predict(self, X):
            return self.fit(X).predict(X)
    
    # ==================== PRINCIPAL COMPONENT ANALYSIS ====================
    
    class PCA:
        """Principal Component Analysis"""
        
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
    
    # ==================== MODEL EVALUATION ====================
    
    class ModelEvaluation:
        """Model evaluation metrics"""
        
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
                classes = np.unique(y_true)
                recalls = []
                for cls in classes:
                    tp = np.sum((y_true == cls) & (y_pred == cls))
                    fn = np.sum((y_true == cls) & (y_pred != cls))
                    recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                return np.mean(recalls)
        
        @staticmethod
        def f1_score(y_true, y_pred, average='binary'):
            prec = MLFromScratch.ModelEvaluation.precision(y_true, y_pred, average)
            rec = MLFromScratch.ModelEvaluation.recall(y_true, y_pred, average)
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
  
  # ==================== DATASET GENERATION ====================
    
    def generate_regression_data(self, n_samples=100, n_features=1, noise=0.1, random_state=42):
        """Generate synthetic regression dataset"""
        np.random.seed(random_state)
        
        X = np.random.randn(n_samples, n_features)
        true_weights = np.random.randn(n_features)
        y = np.dot(X, true_weights) + noise * np.random.randn(n_samples)
        
        return X, y, true_weights
    
    def generate_classification_data(self, n_samples=100, n_features=2, n_classes=2, random_state=42):
        """Generate synthetic classification dataset"""
        np.random.seed(random_state)
        
        X = np.random.randn(n_samples, n_features)
        
        if n_classes == 2:
            # Binary classification
            true_weights = np.random.randn(n_features)
            y_continuous = np.dot(X, true_weights)
            y = (y_continuous > np.median(y_continuous)).astype(int)
        else:
            # Multi-class classification
            centers = np.random.randn(n_classes, n_features) * 2
            y = np.random.randint(0, n_classes, n_samples)
            
            for i in range(n_samples):
                X[i] += centers[y[i]] + 0.5 * np.random.randn(n_features)
        
        return X, y
    
    def generate_clustering_data(self, n_samples=300, n_centers=3, n_features=2, random_state=42):
        """Generate synthetic clustering dataset"""
        np.random.seed(random_state)
        
        centers = np.random.randn(n_centers, n_features) * 3
        X = []
        y = []
        
        samples_per_center = n_samples // n_centers
        
        for i in range(n_centers):
            center_samples = np.random.randn(samples_per_center, n_features) + centers[i]
            X.append(center_samples)
            y.extend([i] * samples_per_center)
        
        X = np.vstack(X)
        y = np.array(y)
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y, centers
    
    # ==================== DEMONSTRATION FUNCTIONS ====================
    
    def demonstrate_linear_regression(self):
        """Demonstrate linear regression"""
        print("\n📈 Linear Regression Demonstration")
        print("=" * 40)
        
        # Generate data
        X, y, true_weights = self.generate_regression_data(n_samples=100, n_features=2, noise=0.1)
        
        # Add bias column
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Train model
        model = self.LinearRegression(learning_rate=0.01, n_iterations=1000, regularization='l2', lambda_reg=0.01)
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate metrics
        mse = np.mean((predictions - y)**2)
        r2 = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))
        
        print(f"True weights: {true_weights}")
        print(f"Learned weights: {model.weights}")
        print(f"Learned bias: {model.bias:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Save results
        results = {
            'model_type': 'Linear Regression',
            'true_weights': true_weights.tolist(),
            'learned_weights': model.weights.tolist(),
            'learned_bias': float(model.bias),
            'mse': float(mse),
            'r2_score': float(r2),
            'cost_history': model.cost_history
        }
        
        with open('results/linear_regression_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Results saved to results/linear_regression_results.json")
        return results
    
    def demonstrate_logistic_regression(self):
        """Demonstrate logistic regression"""
        print("\n🎯 Logistic Regression Demonstration")
        print("=" * 42)
        
        # Generate data
        X, y = self.generate_classification_data(n_samples=200, n_features=2, n_classes=2)
        
        # Train model
        model = self.LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict_classes(X)
        probabilities = model.predict(X)
        
        # Calculate metrics
        accuracy = self.ModelEvaluation.accuracy(y, predictions)
        precision = self.ModelEvaluation.precision(y, predictions)
        recall = self.ModelEvaluation.recall(y, predictions)
        f1 = self.ModelEvaluation.f1_score(y, predictions)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion matrix
        cm = self.ModelEvaluation.confusion_matrix(y, predictions)
        print(f"Confusion Matrix:\n{cm}")
        
        # Save results
        results = {
            'model_type': 'Logistic Regression',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'cost_history': model.cost_history
        }
        
        with open('results/logistic_regression_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Results saved to results/logistic_regression_results.json")
        return results   
 
    def demonstrate_neural_network(self):
        """Demonstrate neural network"""
        print("\n🧠 Neural Network Demonstration")
        print("=" * 35)
        
        # Generate data
        X, y = self.generate_classification_data(n_samples=500, n_features=2, n_classes=3)
        
        # Convert to one-hot encoding for multi-class
        n_classes = len(np.unique(y))
        y_onehot = np.eye(n_classes)[y]
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_onehot[:split_idx], y_onehot[split_idx:]
        y_test_labels = y[split_idx:]
        
        # Train model
        model = self.NeuralNetwork(layers=[2, 10, 5, 3], learning_rate=0.01, activation='relu', optimizer='adam')
        model.fit(X_train, y_train, epochs=500, batch_size=32)
        
        # Make predictions
        predictions = model.predict_classes(X_test)
        
        # Calculate metrics
        accuracy = self.ModelEvaluation.accuracy(y_test_labels, predictions)
        precision = self.ModelEvaluation.precision(y_test_labels, predictions, average='macro')
        recall = self.ModelEvaluation.recall(y_test_labels, predictions, average='macro')
        f1 = self.ModelEvaluation.f1_score(y_test_labels, predictions, average='macro')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        # Confusion matrix
        cm = self.ModelEvaluation.confusion_matrix(y_test_labels, predictions)
        print(f"Confusion Matrix:\n{cm}")
        
        # Save results
        results = {
            'model_type': 'Neural Network',
            'architecture': model.layers,
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'cost_history': model.cost_history
        }
        
        with open('results/neural_network_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Results saved to results/neural_network_results.json")
        return results
    
    def demonstrate_kmeans_clustering(self):
        """Demonstrate K-means clustering"""
        print("\n🎪 K-Means Clustering Demonstration")
        print("=" * 40)
        
        # Generate data
        X, true_labels, true_centers = self.generate_clustering_data(n_samples=300, n_centers=4, n_features=2)
        
        # Train model
        model = self.KMeans(k=4, max_iters=100, random_state=42)
        predicted_labels = model.fit_predict(X)
        
        # Calculate metrics (using true labels for evaluation)
        # Note: K-means labels might not match true labels directly
        def cluster_accuracy(true_labels, pred_labels):
            from scipy.optimize import linear_sum_assignment
            # This is a simplified version - in practice you'd use Hungarian algorithm
            unique_true = np.unique(true_labels)
            unique_pred = np.unique(pred_labels)
            
            # Simple matching based on majority vote
            best_accuracy = 0
            for perm in np.random.permutation(unique_pred)[:10]:  # Try a few permutations
                mapped_pred = pred_labels.copy()
                for i, p in enumerate(perm[:len(unique_true)]):
                    mapped_pred[pred_labels == p] = i
                accuracy = np.mean(true_labels == mapped_pred)
                best_accuracy = max(best_accuracy, accuracy)
            
            return best_accuracy
        
        accuracy = cluster_accuracy(true_labels, predicted_labels)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = 0
        for i in range(model.k):
            cluster_points = X[predicted_labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - model.centroids[i])**2)
        
        print(f"Clustering Accuracy: {accuracy:.4f}")
        print(f"Inertia (WCSS): {inertia:.4f}")
        print(f"Number of clusters: {model.k}")
        
        # Save results
        results = {
            'model_type': 'K-Means Clustering',
            'n_clusters': int(model.k),
            'clustering_accuracy': float(accuracy),
            'inertia': float(inertia),
            'centroids': model.centroids.tolist(),
            'true_centers': true_centers.tolist()
        }
        
        with open('results/kmeans_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Results saved to results/kmeans_results.json")
        return results
    
    def demonstrate_pca(self):
        """Demonstrate Principal Component Analysis"""
        print("\n📊 PCA Demonstration")
        print("=" * 22)
        
        # Generate high-dimensional data
        np.random.seed(42)
        n_samples, n_features = 200, 10
        
        # Create correlated features
        true_components = np.random.randn(3, n_features)
        coefficients = np.random.randn(n_samples, 3)
        X = np.dot(coefficients, true_components) + 0.1 * np.random.randn(n_samples, n_features)
        
        # Apply PCA
        model = self.PCA(n_components=3)
        X_transformed = model.fit_transform(X)
        
        # Calculate metrics
        total_variance_explained = np.sum(model.explained_variance_ratio_)
        
        print(f"Original dimensions: {X.shape[1]}")
        print(f"Reduced dimensions: {X_transformed.shape[1]}")
        print(f"Explained variance ratio: {model.explained_variance_ratio_}")
        print(f"Total variance explained: {total_variance_explained:.4f}")
        
        # Reconstruction error
        X_reconstructed = np.dot(X_transformed, model.components_.T) + model.mean_
        reconstruction_error = np.mean((X - X_reconstructed)**2)
        print(f"Reconstruction error: {reconstruction_error:.4f}")
        
        # Save results
        results = {
            'model_type': 'PCA',
            'original_dimensions': int(X.shape[1]),
            'reduced_dimensions': int(X_transformed.shape[1]),
            'explained_variance_ratio': model.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(total_variance_explained),
            'reconstruction_error': float(reconstruction_error)
        }
        
        with open('results/pca_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Results saved to results/pca_results.json")
        return results
    
    def run_all_demonstrations(self):
        """Run all ML demonstrations"""
        print("🤖 MACHINE LEARNING FROM SCRATCH")
        print("=" * 40)
        print("Complete ML algorithms using only NumPy")
        print("Advanced linear algebra and optimization\n")
        
        # Run all demonstrations
        demos = [
            self.demonstrate_linear_regression,
            self.demonstrate_logistic_regression,
            self.demonstrate_neural_network,
            self.demonstrate_kmeans_clustering,
            self.demonstrate_pca
        ]
        
        results = {}
        
        for demo in demos:
            try:
                result = demo()
                results[demo.__name__] = result
            except Exception as e:
                print(f"❌ Error in {demo.__name__}: {e}")
                continue
        
        print("\n✅ ALL ML DEMONSTRATIONS COMPLETE!")
        print("Key NumPy concepts mastered:")
        print("- Advanced linear algebra operations")
        print("- Gradient computation and backpropagation")
        print("- Matrix factorization techniques")
        print("- Broadcasting for batch operations")
        print("- Memory-efficient array operations")
        print("- Optimization algorithms implementation")
        print("- Statistical analysis and evaluation")
        print("- Eigenvalue decomposition and PCA")
        
        return results

def main():
    """Main function to run ML demonstrations"""
    # Create ML framework
    ml_framework = MLFromScratch()
    
    # Run all demonstrations
    results = ml_framework.run_all_demonstrations()
    
    print(f"\n📁 All results saved in 'results/' directory")
    print("🎬 Ready for Streamlit visualization!")

if __name__ == "__main__":
    main()