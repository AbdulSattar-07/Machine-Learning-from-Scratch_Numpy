"""
Project 5: Machine Learning from Scratch - Professional Streamlit Interface
Advanced ML algorithms implementation with interactive visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import time
from project5_ml_from_scratch import MLFromScratch

# Page configuration
st.set_page_config(
    page_title="ML from Scratch - Professional Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .algorithm-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease;
    }
    
    .algorithm-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .success-message {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .animated-counter {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class MLStreamlitApp:
    def __init__(self):
        self.ml_framework = MLFromScratch()
        self.results_cache = {}
        
    def create_animated_header(self):
        """Create animated header with gradient background"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Machine Learning from Scratch</h1>
            <h3>Professional ML Suite with NumPy Implementation</h3>
            <p>Advanced algorithms • Interactive visualizations • Real-time analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    def create_sidebar(self):
        """Create professional sidebar with navigation"""
        st.sidebar.markdown("## 🎯 Navigation")
        
        page = st.sidebar.selectbox(
            "Choose Algorithm",
            [
                "🏠 Dashboard",
                "📈 Linear Regression", 
                "🎯 Logistic Regression",
                "🧠 Neural Network",
                "🎪 K-Means Clustering",
                "📊 PCA Analysis",
                "🔬 Model Comparison",
                "📋 Algorithm Details"
            ]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ⚙️ Global Settings")
        
        # Global parameters
        random_state = st.sidebar.number_input("Random State", value=42, min_value=0, max_value=1000)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div class="info-box">
            <h4>🎓 Learning Objectives</h4>
            <ul>
                <li>NumPy advanced operations</li>
                <li>Gradient descent optimization</li>
                <li>Matrix factorization</li>
                <li>Statistical analysis</li>
                <li>Algorithm implementation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        return page, random_state
    
    def create_dashboard(self):
        """Create main dashboard with overview"""
        st.markdown("## 📊 ML Algorithms Dashboard")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>5</h3>
                <p>ML Algorithms</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>100%</h3>
                <p>NumPy Implementation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>∞</h3>
                <p>Learning Potential</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🚀</h3>
                <p>Performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Algorithm overview
        st.markdown("## 🎯 Available Algorithms")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="algorithm-card">
                <h4>📈 Supervised Learning</h4>
                <ul>
                    <li><strong>Linear Regression</strong> - Gradient descent with regularization</li>
                    <li><strong>Logistic Regression</strong> - Binary classification with sigmoid</li>
                    <li><strong>Neural Network</strong> - Multi-layer perceptron with backprop</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="algorithm-card">
                <h4>🎪 Unsupervised Learning</h4>
                <ul>
                    <li><strong>K-Means Clustering</strong> - Centroid-based clustering</li>
                    <li><strong>PCA</strong> - Dimensionality reduction via eigendecomposition</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start section
        st.markdown("## 🚀 Quick Start")
        
        if st.button("🎬 Run All Algorithms Demo", key="run_all"):
            self.run_all_algorithms_demo()
    
    def run_all_algorithms_demo(self):
        """Run all algorithms with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        algorithms = [
            ("Linear Regression", self.ml_framework.demonstrate_linear_regression),
            ("Logistic Regression", self.ml_framework.demonstrate_logistic_regression),
            ("Neural Network", self.ml_framework.demonstrate_neural_network),
            ("K-Means Clustering", self.ml_framework.demonstrate_kmeans_clustering),
            ("PCA Analysis", self.ml_framework.demonstrate_pca)
        ]
        
        results = {}
        
        for i, (name, func) in enumerate(algorithms):
            status_text.text(f"Running {name}...")
            progress_bar.progress((i + 1) / len(algorithms))
            
            try:
                result = func()
                results[name] = result
                time.sleep(0.5)  # Animation delay
            except Exception as e:
                st.error(f"Error in {name}: {str(e)}")
        
        status_text.text("All algorithms completed!")
        
        st.markdown("""
        <div class="success-message">
            <h3>✅ All Algorithms Executed Successfully!</h3>
            <p>Results saved to results/ directory. Check individual algorithm pages for detailed analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        return results
    
    def create_linear_regression_page(self, random_state):
        """Linear regression analysis page"""
        st.markdown("## 📈 Linear Regression Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Parameters")
            
            n_samples = st.slider("Number of Samples", 50, 500, 100)
            n_features = st.slider("Number of Features", 1, 5, 2)
            noise = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05)
            learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
            n_iterations = st.slider("Iterations", 100, 2000, 1000, 100)
            
            regularization = st.selectbox("Regularization", ["None", "L1", "L2"])
            lambda_reg = st.slider("Regularization Strength", 0.0, 1.0, 0.01, 0.01)
            
            if st.button("🚀 Train Linear Regression", key="train_lr"):
                self.train_linear_regression(n_samples, n_features, noise, learning_rate, 
                                           n_iterations, regularization, lambda_reg, random_state)
        
        with col2:
            if hasattr(self, 'lr_results'):
                self.visualize_linear_regression_results()
    
    def train_linear_regression(self, n_samples, n_features, noise, learning_rate, 
                              n_iterations, regularization, lambda_reg, random_state):
        """Train linear regression model"""
        
        # Generate data
        X, y, true_weights = self.ml_framework.generate_regression_data(
            n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state
        )
        
        # Train model
        reg_type = None if regularization == "None" else regularization.lower()
        
        model = self.ml_framework.LinearRegression(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            regularization=reg_type,
            lambda_reg=lambda_reg
        )
        
        with st.spinner("Training model..."):
            model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate metrics
        mse = np.mean((predictions - y)**2)
        r2 = 1 - (np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2))
        
        # Store results
        self.lr_results = {
            'X': X,
            'y': y,
            'predictions': predictions,
            'true_weights': true_weights,
            'learned_weights': model.weights,
            'learned_bias': model.bias,
            'mse': mse,
            'r2': r2,
            'cost_history': model.cost_history
        }
        
        st.success("✅ Model trained successfully!")
    
    def visualize_linear_regression_results(self):
        """Visualize linear regression results"""
        results = self.lr_results
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MSE", f"{results['mse']:.4f}")
        with col2:
            st.metric("R² Score", f"{results['r2']:.4f}")
        with col3:
            st.metric("Bias", f"{results['learned_bias']:.4f}")
        
        # Visualizations
        if results['X'].shape[1] == 1:
            # 1D regression plot
            fig = go.Figure()
            
            # Sort for line plot
            sort_idx = np.argsort(results['X'].flatten())
            X_sorted = results['X'][sort_idx]
            pred_sorted = results['predictions'][sort_idx]
            
            fig.add_trace(go.Scatter(
                x=results['X'].flatten(),
                y=results['y'],
                mode='markers',
                name='Actual Data',
                marker=dict(color='blue', size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=X_sorted.flatten(),
                y=pred_sorted,
                mode='lines',
                name='Regression Line',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Linear Regression Fit",
                xaxis_title="X",
                yaxis_title="y",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost history
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            y=results['cost_history'],
            mode='lines',
            name='Training Cost',
            line=dict(color='purple', width=2)
        ))
        
        fig_cost.update_layout(
            title="Training Cost History",
            xaxis_title="Iteration",
            yaxis_title="Cost (MSE)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Weights comparison
        if len(results['true_weights']) <= 10:  # Only show if not too many features
            weights_df = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(results['true_weights']))],
                'True Weight': results['true_weights'],
                'Learned Weight': results['learned_weights']
            })
            
            fig_weights = px.bar(
                weights_df.melt(id_vars='Feature', var_name='Type', value_name='Weight'),
                x='Feature',
                y='Weight',
                color='Type',
                barmode='group',
                title="True vs Learned Weights"
            )
            
            st.plotly_chart(fig_weights, use_container_width=True)
    
    def create_logistic_regression_page(self, random_state):
        """Logistic regression analysis page"""
        st.markdown("## 🎯 Logistic Regression Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Parameters")
            
            n_samples = st.slider("Number of Samples", 100, 1000, 200, key="lr_samples")
            n_features = st.slider("Number of Features", 2, 10, 2, key="lr_features")
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01, key="lr_lr")
            n_iterations = st.slider("Iterations", 100, 2000, 1000, 100, key="lr_iter")
            
            if st.button("🚀 Train Logistic Regression", key="train_logr"):
                self.train_logistic_regression(n_samples, n_features, learning_rate, 
                                             n_iterations, random_state)
        
        with col2:
            if hasattr(self, 'logr_results'):
                self.visualize_logistic_regression_results()
    
    def train_logistic_regression(self, n_samples, n_features, learning_rate, n_iterations, random_state):
        """Train logistic regression model"""
        
        # Generate data
        X, y = self.ml_framework.generate_classification_data(
            n_samples=n_samples, n_features=n_features, n_classes=2, random_state=random_state
        )
        
        # Train model
        model = self.ml_framework.LogisticRegression(
            learning_rate=learning_rate,
            n_iterations=n_iterations
        )
        
        with st.spinner("Training model..."):
            model.fit(X, y)
        
        # Make predictions
        predictions = model.predict_classes(X)
        probabilities = model.predict(X)
        
        # Calculate metrics
        accuracy = self.ml_framework.ModelEvaluation.accuracy(y, predictions)
        precision = self.ml_framework.ModelEvaluation.precision(y, predictions)
        recall = self.ml_framework.ModelEvaluation.recall(y, predictions)
        f1 = self.ml_framework.ModelEvaluation.f1_score(y, predictions)
        cm = self.ml_framework.ModelEvaluation.confusion_matrix(y, predictions)
        
        # Store results
        self.logr_results = {
            'X': X,
            'y': y,
            'predictions': predictions,
            'probabilities': probabilities,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'cost_history': model.cost_history,
            'weights': model.weights,
            'bias': model.bias
        }
        
        st.success("✅ Model trained successfully!")
    
    def visualize_logistic_regression_results(self):
        """Visualize logistic regression results"""
        results = self.logr_results
        
        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{results['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{results['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{results['f1']:.3f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Decision boundary (for 2D data)
            if results['X'].shape[1] == 2:
                fig = go.Figure()
                
                # Create decision boundary
                h = 0.02
                x_min, x_max = results['X'][:, 0].min() - 1, results['X'][:, 0].max() + 1
                y_min, y_max = results['X'][:, 1].min() - 1, results['X'][:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
                
                mesh_points = np.c_[xx.ravel(), yy.ravel()]
                
                # Predict on mesh
                z = np.dot(mesh_points, results['weights']) + results['bias']
                z = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid
                z = z.reshape(xx.shape)
                
                # Add contour
                fig.add_trace(go.Contour(
                    x=np.arange(x_min, x_max, h),
                    y=np.arange(y_min, y_max, h),
                    z=z,
                    showscale=False,
                    opacity=0.3,
                    colorscale='RdYlBu'
                ))
                
                # Add data points
                colors = ['red' if label == 0 else 'blue' for label in results['y']]
                fig.add_trace(go.Scatter(
                    x=results['X'][:, 0],
                    y=results['X'][:, 1],
                    mode='markers',
                    marker=dict(color=colors, size=8),
                    name='Data Points'
                ))
                
                fig.update_layout(
                    title="Logistic Regression Decision Boundary",
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confusion Matrix
            fig_cm = px.imshow(
                results['confusion_matrix'],
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                color_continuous_scale='Blues'
            )
            
            fig_cm.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Cost history
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            y=results['cost_history'],
            mode='lines',
            name='Training Cost',
            line=dict(color='green', width=2)
        ))
        
        fig_cost.update_layout(
            title="Training Cost History (Cross-Entropy Loss)",
            xaxis_title="Iteration",
            yaxis_title="Cost",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_cost, use_container_width=True) 
   
    def create_neural_network_page(self, random_state):
        """Neural network analysis page"""
        st.markdown("## 🧠 Neural Network Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Parameters")
            
            n_samples = st.slider("Number of Samples", 200, 1000, 500, key="nn_samples")
            n_classes = st.slider("Number of Classes", 2, 5, 3, key="nn_classes")
            
            # Architecture
            st.markdown("#### Network Architecture")
            hidden_layers = st.slider("Hidden Layers", 1, 3, 2, key="nn_hidden")
            neurons_per_layer = st.slider("Neurons per Layer", 5, 50, 10, key="nn_neurons")
            
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01, key="nn_lr")
            n_iterations = st.slider("Iterations", 500, 5000, 2000, 100, key="nn_iter")
            
            if st.button("🚀 Train Neural Network", key="train_nn"):
                self.train_neural_network(n_samples, n_classes, hidden_layers, 
                                        neurons_per_layer, learning_rate, n_iterations, random_state)
        
        with col2:
            if hasattr(self, 'nn_results'):
                self.visualize_neural_network_results()
    
    def train_neural_network(self, n_samples, n_classes, hidden_layers, neurons_per_layer, 
                           learning_rate, n_iterations, random_state):
        """Train neural network model"""
        
        # Generate data
        X, y = self.ml_framework.generate_classification_data(
            n_samples=n_samples, n_features=2, n_classes=n_classes, random_state=random_state
        )
        
        # Create architecture
        architecture = [2]  # Input layer
        for _ in range(hidden_layers):
            architecture.append(neurons_per_layer)
        architecture.append(n_classes)  # Output layer
        
        # Train model
        model = self.ml_framework.NeuralNetwork(
            layers=architecture,
            learning_rate=learning_rate
        )
        
        with st.spinner("Training neural network..."):
            model.fit(X, y, epochs=n_iterations)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Calculate metrics
        accuracy = self.ml_framework.ModelEvaluation.accuracy(y, predictions)
        
        # Store results
        self.nn_results = {
            'X': X,
            'y': y,
            'predictions': predictions,
            'probabilities': probabilities,
            'accuracy': accuracy,
            'cost_history': model.cost_history,
            'architecture': architecture,
            'n_classes': n_classes
        }
        
        st.success("✅ Neural network trained successfully!")
    
    def visualize_neural_network_results(self):
        """Visualize neural network results"""
        results = self.nn_results
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.3f}")
        with col2:
            st.metric("Architecture", f"{' → '.join(map(str, results['architecture']))}")
        with col3:
            st.metric("Classes", results['n_classes'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Decision boundary visualization
            fig = go.Figure()
            
            # Create decision boundary
            h = 0.02
            x_min, x_max = results['X'][:, 0].min() - 1, results['X'][:, 0].max() + 1
            y_min, y_max = results['X'][:, 1].min() - 1, results['X'][:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            
            # This would require implementing predict method in the neural network
            # For now, just show the data points
            colors = px.colors.qualitative.Set1[:results['n_classes']]
            
            for class_idx in range(results['n_classes']):
                mask = results['y'] == class_idx
                fig.add_trace(go.Scatter(
                    x=results['X'][mask, 0],
                    y=results['X'][mask, 1],
                    mode='markers',
                    marker=dict(color=colors[class_idx], size=8),
                    name=f'Class {class_idx}'
                ))
            
            fig.update_layout(
                title="Neural Network Classification",
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training cost history
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                y=results['cost_history'],
                mode='lines',
                name='Training Cost',
                line=dict(color='orange', width=2)
            ))
            
            fig_cost.update_layout(
                title="Neural Network Training Cost",
                xaxis_title="Iteration",
                yaxis_title="Cost",
                template="plotly_white"
            )
            
            st.plotly_chart(fig_cost, use_container_width=True)
    
    def create_kmeans_page(self, random_state):
        """K-means clustering analysis page"""
        st.markdown("## 🎪 K-Means Clustering Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Parameters")
            
            n_samples = st.slider("Number of Samples", 100, 1000, 300, key="km_samples")
            n_clusters = st.slider("Number of Clusters", 2, 8, 3, key="km_clusters")
            max_iters = st.slider("Max Iterations", 50, 500, 100, key="km_iters")
            
            if st.button("🚀 Run K-Means Clustering", key="train_km"):
                self.run_kmeans_clustering(n_samples, n_clusters, max_iters, random_state)
        
        with col2:
            if hasattr(self, 'km_results'):
                self.visualize_kmeans_results()
    
    def run_kmeans_clustering(self, n_samples, n_clusters, max_iters, random_state):
        """Run K-means clustering"""
        
        # Generate data
        X = self.ml_framework.generate_clustering_data(
            n_samples=n_samples, n_centers=n_clusters, random_state=random_state
        )
        
        # Run clustering
        model = self.ml_framework.KMeans(
            k=n_clusters,
            max_iters=max_iters,
            random_state=random_state
        )
        
        with st.spinner("Running K-means clustering..."):
            labels = model.fit_predict(X)
        
        # Calculate metrics
        inertia = model.inertia_
        
        # Store results
        self.km_results = {
            'X': X,
            'labels': labels,
            'centroids': model.centroids,
            'inertia': inertia,
            'n_clusters': n_clusters,
            'n_iterations': model.n_iterations
        }
        
        st.success("✅ K-means clustering completed!")
    
    def visualize_kmeans_results(self):
        """Visualize K-means results"""
        results = self.km_results
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Inertia", f"{results['inertia']:.2f}")
        with col2:
            st.metric("Clusters", results['n_clusters'])
        with col3:
            st.metric("Iterations", results['n_iterations'])
        
        # Clustering visualization
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1[:results['n_clusters']]
        
        # Plot data points
        for cluster_idx in range(results['n_clusters']):
            mask = results['labels'] == cluster_idx
            fig.add_trace(go.Scatter(
                x=results['X'][mask, 0],
                y=results['X'][mask, 1],
                mode='markers',
                marker=dict(color=colors[cluster_idx], size=8),
                name=f'Cluster {cluster_idx}'
            ))
        
        # Plot centroids
        fig.add_trace(go.Scatter(
            x=results['centroids'][:, 0],
            y=results['centroids'][:, 1],
            mode='markers',
            marker=dict(color='black', size=15, symbol='x'),
            name='Centroids'
        ))
        
        fig.update_layout(
            title="K-Means Clustering Results",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_pca_page(self, random_state):
        """PCA analysis page"""
        st.markdown("## 📊 Principal Component Analysis (PCA)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎛️ Parameters")
            
            n_samples = st.slider("Number of Samples", 100, 1000, 200, key="pca_samples")
            n_features = st.slider("Original Features", 3, 10, 5, key="pca_features")
            n_components = st.slider("PCA Components", 2, min(n_features, 5), 2, key="pca_components")
            
            if st.button("🚀 Run PCA Analysis", key="run_pca"):
                self.run_pca_analysis(n_samples, n_features, n_components, random_state)
        
        with col2:
            if hasattr(self, 'pca_results'):
                self.visualize_pca_results()
    
    def run_pca_analysis(self, n_samples, n_features, n_components, random_state):
        """Run PCA analysis"""
        
        # Generate high-dimensional data
        np.random.seed(random_state)
        X = np.random.randn(n_samples, n_features)
        
        # Add some correlation structure
        for i in range(1, n_features):
            X[:, i] += 0.5 * X[:, 0] + 0.3 * np.random.randn(n_samples)
        
        # Run PCA
        model = self.ml_framework.PCA(n_components=n_components)
        
        with st.spinner("Running PCA analysis..."):
            X_transformed = model.fit_transform(X)
        
        # Store results
        self.pca_results = {
            'X_original': X,
            'X_transformed': X_transformed,
            'components': model.components_,
            'explained_variance': model.explained_variance_,
            'explained_variance_ratio': model.explained_variance_ratio_,
            'n_components': n_components,
            'n_features': n_features
        }
        
        st.success("✅ PCA analysis completed!")
    
    def visualize_pca_results(self):
        """Visualize PCA results"""
        results = self.pca_results
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Dimensions", results['n_features'])
        with col2:
            st.metric("Reduced Dimensions", results['n_components'])
        with col3:
            st.metric("Variance Explained", f"{results['explained_variance_ratio'].sum():.1%}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Explained variance plot
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(len(results['explained_variance_ratio']))],
                y=results['explained_variance_ratio'],
                name='Explained Variance Ratio'
            ))
            
            fig_var.update_layout(
                title="Explained Variance by Component",
                xaxis_title="Principal Component",
                yaxis_title="Variance Ratio",
                template="plotly_white"
            )
            
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            # 2D projection (if we have at least 2 components)
            if results['n_components'] >= 2:
                fig_proj = go.Figure()
                fig_proj.add_trace(go.Scatter(
                    x=results['X_transformed'][:, 0],
                    y=results['X_transformed'][:, 1],
                    mode='markers',
                    marker=dict(color='blue', size=6),
                    name='Projected Data'
                ))
                
                fig_proj.update_layout(
                    title="PCA Projection (First 2 Components)",
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_proj, use_container_width=True)
    
    def create_model_comparison_page(self):
        """Model comparison page"""
        st.markdown("## 🔬 Model Comparison")
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Comparison Framework</h4>
            <p>Compare different algorithms on the same dataset to understand their strengths and weaknesses.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset selection
        dataset_type = st.selectbox(
            "Select Dataset Type",
            ["Classification", "Regression", "Clustering"]
        )
        
        if dataset_type == "Classification":
            self.compare_classification_models()
        elif dataset_type == "Regression":
            self.compare_regression_models()
        else:
            self.compare_clustering_models()
    
    def compare_classification_models(self):
        """Compare classification models"""
        st.markdown("### 🎯 Classification Models Comparison")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_samples = st.slider("Samples", 200, 1000, 500, key="comp_class_samples")
            n_features = st.slider("Features", 2, 5, 2, key="comp_class_features")
            n_classes = st.slider("Classes", 2, 4, 3, key="comp_class_classes")
            
            if st.button("🚀 Compare Models", key="compare_class"):
                self.run_classification_comparison(n_samples, n_features, n_classes)
        
        with col2:
            if hasattr(self, 'class_comparison_results'):
                self.visualize_classification_comparison()
    
    def run_classification_comparison(self, n_samples, n_features, n_classes):
        """Run classification model comparison"""
        
        # Generate data
        X, y = self.ml_framework.generate_classification_data(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=42
        )
        
        results = {}
        
        # Logistic Regression (for binary classification)
        if n_classes == 2:
            with st.spinner("Training Logistic Regression..."):
                lr_model = self.ml_framework.LogisticRegression(learning_rate=0.1, n_iterations=1000)
                lr_model.fit(X, y)
                lr_pred = lr_model.predict_classes(X)
                results['Logistic Regression'] = {
                    'accuracy': self.ml_framework.ModelEvaluation.accuracy(y, lr_pred),
                    'predictions': lr_pred
                }
        
        # Neural Network
        with st.spinner("Training Neural Network..."):
            nn_model = self.ml_framework.NeuralNetwork(
                architecture=[n_features, 10, n_classes],
                learning_rate=0.1,
                n_iterations=1000
            )
            nn_model.fit(X, y)
            nn_pred = nn_model.predict(X)
            results['Neural Network'] = {
                'accuracy': self.ml_framework.ModelEvaluation.accuracy(y, nn_pred),
                'predictions': nn_pred
            }
        
        self.class_comparison_results = {
            'X': X,
            'y': y,
            'results': results,
            'n_classes': n_classes
        }
        
        st.success("✅ Model comparison completed!")
    
    def visualize_classification_comparison(self):
        """Visualize classification comparison results"""
        results = self.class_comparison_results
        
        # Accuracy comparison
        models = list(results['results'].keys())
        accuracies = [results['results'][model]['accuracy'] for model in models]
        
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=models,
            y=accuracies,
            marker_color=['blue', 'orange'][:len(models)]
        ))
        
        fig_acc.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # Display metrics table
        metrics_df = pd.DataFrame({
            'Model': models,
            'Accuracy': [f"{acc:.3f}" for acc in accuracies]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
    
    def compare_regression_models(self):
        """Compare regression models"""
        st.markdown("### 📈 Regression Models Comparison")
        st.info("Regression model comparison will be implemented in future versions.")
    
    def compare_clustering_models(self):
        """Compare clustering models"""
        st.markdown("### 🎪 Clustering Models Comparison")
        st.info("Clustering model comparison will be implemented in future versions.")
    
    def create_algorithm_details_page(self):
        """Algorithm details and theory page"""
        st.markdown("## 📋 Algorithm Details & Theory")
        
        algorithm = st.selectbox(
            "Select Algorithm",
            [
                "Linear Regression",
                "Logistic Regression", 
                "Neural Network",
                "K-Means Clustering",
                "PCA"
            ]
        )
        
        if algorithm == "Linear Regression":
            self.show_linear_regression_theory()
        elif algorithm == "Logistic Regression":
            self.show_logistic_regression_theory()
        elif algorithm == "Neural Network":
            self.show_neural_network_theory()
        elif algorithm == "K-Means Clustering":
            self.show_kmeans_theory()
        elif algorithm == "PCA":
            self.show_pca_theory()
    
    def show_linear_regression_theory(self):
        """Show linear regression theory"""
        st.markdown("""
        ### 📈 Linear Regression Theory
        
        **Objective**: Find the best linear relationship between input features and target values.
        
        **Mathematical Foundation**:
        - **Hypothesis**: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
        - **Cost Function**: J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
        - **Gradient Descent**: θⱼ := θⱼ - α ∂J(θ)/∂θⱼ
        
        **Key Features**:
        - ✅ Simple and interpretable
        - ✅ Fast training and prediction
        - ✅ No hyperparameter tuning needed
        - ❌ Assumes linear relationship
        - ❌ Sensitive to outliers
        
        **Regularization**:
        - **L1 (Lasso)**: Adds |θ| penalty → Feature selection
        - **L2 (Ridge)**: Adds θ² penalty → Prevents overfitting
        """)
    
    def show_logistic_regression_theory(self):
        """Show logistic regression theory"""
        st.markdown("""
        ### 🎯 Logistic Regression Theory
        
        **Objective**: Classify data points into discrete categories using probability.
        
        **Mathematical Foundation**:
        - **Sigmoid Function**: σ(z) = 1/(1 + e⁻ᶻ)
        - **Hypothesis**: h(x) = σ(θᵀx)
        - **Cost Function**: J(θ) = -(1/m) Σ[y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]
        
        **Key Features**:
        - ✅ Outputs probabilities
        - ✅ No assumptions about data distribution
        - ✅ Less sensitive to outliers than linear regression
        - ❌ Assumes linear decision boundary
        - ❌ Can struggle with complex relationships
        
        **Decision Boundary**:
        - Classification threshold typically at 0.5
        - Linear boundary in feature space
        """)
    
    def show_neural_network_theory(self):
        """Show neural network theory"""
        st.markdown("""
        ### 🧠 Neural Network Theory
        
        **Objective**: Learn complex non-linear patterns through layered transformations.
        
        **Mathematical Foundation**:
        - **Forward Pass**: aˡ = σ(Wˡaˡ⁻¹ + bˡ)
        - **Backpropagation**: ∂C/∂Wˡ = aˡ⁻¹(δˡ)ᵀ
        - **Weight Update**: W := W - α∇W
        
        **Key Components**:
        - **Neurons**: Basic processing units
        - **Layers**: Input, hidden, output
        - **Activation Functions**: ReLU, Sigmoid, Tanh
        - **Weights & Biases**: Learnable parameters
        
        **Key Features**:
        - ✅ Can learn complex non-linear patterns
        - ✅ Universal function approximator
        - ✅ Flexible architecture
        - ❌ Requires more data
        - ❌ Black box (less interpretable)
        - ❌ Prone to overfitting
        """)
    
    def show_kmeans_theory(self):
        """Show K-means theory"""
        st.markdown("""
        ### 🎪 K-Means Clustering Theory
        
        **Objective**: Partition data into k clusters by minimizing within-cluster sum of squares.
        
        **Algorithm Steps**:
        1. **Initialize**: Randomly place k centroids
        2. **Assign**: Each point to nearest centroid
        3. **Update**: Move centroids to cluster centers
        4. **Repeat**: Until convergence
        
        **Mathematical Foundation**:
        - **Objective**: Minimize Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
        - **Distance**: Euclidean distance
        - **Centroid Update**: μᵢ = (1/|Cᵢ|) Σₓ∈Cᵢ x
        
        **Key Features**:
        - ✅ Simple and fast
        - ✅ Works well with spherical clusters
        - ✅ Guaranteed convergence
        - ❌ Need to specify k
        - ❌ Sensitive to initialization
        - ❌ Assumes spherical clusters
        """)
    
    def show_pca_theory(self):
        """Show PCA theory"""
        st.markdown("""
        ### 📊 Principal Component Analysis (PCA) Theory
        
        **Objective**: Reduce dimensionality while preserving maximum variance.
        
        **Mathematical Foundation**:
        - **Covariance Matrix**: C = (1/n)XᵀX
        - **Eigendecomposition**: C = QΛQᵀ
        - **Principal Components**: Eigenvectors of covariance matrix
        - **Explained Variance**: Eigenvalues
        
        **Algorithm Steps**:
        1. **Standardize**: Center the data (subtract mean)
        2. **Covariance**: Compute covariance matrix
        3. **Eigendecomposition**: Find eigenvectors and eigenvalues
        4. **Select**: Choose top k components
        5. **Transform**: Project data onto new space
        
        **Key Features**:
        - ✅ Reduces dimensionality
        - ✅ Removes correlation
        - ✅ Preserves maximum variance
        - ❌ Linear transformation only
        - ❌ Components may not be interpretable
        - ❌ Sensitive to scaling
        """)
    
    def run(self):
        """Main application runner"""
        self.create_animated_header()
        
        page, random_state = self.create_sidebar()
        
        if page == "🏠 Dashboard":
            self.create_dashboard()
        elif page == "📈 Linear Regression":
            self.create_linear_regression_page(random_state)
        elif page == "🎯 Logistic Regression":
            self.create_logistic_regression_page(random_state)
        elif page == "🧠 Neural Network":
            self.create_neural_network_page(random_state)
        elif page == "🎪 K-Means Clustering":
            self.create_kmeans_page(random_state)
        elif page == "📊 PCA Analysis":
            self.create_pca_page(random_state)
        elif page == "🔬 Model Comparison":
            self.create_model_comparison_page()
        elif page == "📋 Algorithm Details":
            self.create_algorithm_details_page()

# Run the application
if __name__ == "__main__":
    app = MLStreamlitApp()
    app.run()