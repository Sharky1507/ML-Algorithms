# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.datasets import make_classification, make_blobs
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Explorer Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
    }
    .algorithm-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .feature-importance {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class MLPlayground:
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.results = {}
        
    def generate_synthetic_data(self, data_type, n_samples=1000):
        """Generate various types of synthetic datasets"""
        np.random.seed(42)
        
        if data_type == "Classification":
            X, y = make_classification(
                n_samples=n_samples, n_features=10, n_informative=6,
                n_redundant=2, n_clusters_per_class=1, random_state=42
            )
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            
        elif data_type == "Clustering":
            X, y = make_blobs(
                n_samples=n_samples, n_features=8, 
                centers=4, cluster_std=1.5, random_state=42
            )
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            
        elif data_type == "Anomaly Detection":
            X, y = make_blobs(
                n_samples=n_samples, n_features=6, 
                centers=1, cluster_std=1.0, random_state=42
            )
            # Add some outliers
            outliers = np.random.uniform(low=-10, high=10, size=(20, 6))
            X = np.vstack([X, outliers])
            y = np.hstack([np.zeros(n_samples), np.ones(20)])  # 1 for anomalies
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            
        return X, y, feature_names
    
    def train_model(self, model_type, X_train, y_train, **params):
        """Train different ML models"""
        if model_type == "Random Forest":
            model = RandomForestClassifier(**params, random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(**params, random_state=42)
        elif model_type == "SVM":
            model = SVC(**params, probability=True, random_state=42)
        elif model_type == "K-Means":
            model = KMeans(**params, random_state=42)
        elif model_type == "DBSCAN":
            model = DBSCAN(**params)
        elif model_type == "Gaussian Mixture":
            model = GaussianMixture(**params, random_state=42)
        elif model_type == "Isolation Forest":
            model = IsolationForest(**params, random_state=42, contamination=0.1)
        
        if model_type in ["K-Means", "DBSCAN", "Gaussian Mixture", "Isolation Forest"]:
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)
            
        return model
    
    def perform_feature_selection(self, X, y, method='kbest', k=5):
        """Perform feature selection using different methods"""
        if method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            scores = selector.scores_
            return X_selected, scores, selector.get_support()
        
        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            X_selected = X[:, indices]
            return X_selected, importances, indices

def main():
    st.markdown('<h1 class="main-header">🔍 ML Explorer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Machine Learning Playground & Model Benchmarking")
    
    if 'playground' not in st.session_state:
        # Initialize playground
        st.session_state.playground = MLPlayground()
        st.session_state.dataset_ready = False
        
    playground = st.session_state.playground
    
    # Ensure state consistency
    if st.session_state.get('dataset_ready', False) and 'current' not in playground.datasets:
        st.session_state.dataset_ready = False
    
    # Sidebar
    st.sidebar.title("Configuration")
    app_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["📊 Data Explorer", "🎯 Model Trainer", "🔍 Feature Analysis", "📈 Model Benchmark"]
    )
    
    # Main content
    if app_mode == "📊 Data Explorer":
        data_explorer(playground)
    elif app_mode == "🎯 Model Trainer":
        model_trainer(playground)
    elif app_mode == "🔍 Feature Analysis":
        feature_analysis(playground)
    else:
        model_benchmark(playground)

def data_explorer(playground):
    st.header("📊 Interactive Data Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Dataset Configuration")
        data_type = st.selectbox(
            "Select Data Type",
            ["Classification", "Clustering", "Anomaly Detection"]
        )
        
        n_samples = st.slider("Number of Samples", 100, 5000, 1000)
        n_features = st.slider("Number of Features", 5, 20, 10)
        
        if st.button("Generate Dataset", use_container_width=True):
            with st.spinner("Generating synthetic dataset..."):
                X, y, feature_names = playground.generate_synthetic_data(data_type, n_samples)
                playground.datasets['current'] = {'X': X, 'y': y, 'feature_names': feature_names}
                st.session_state.dataset_ready = True
                st.rerun()
    with col2:
        if 'dataset_ready' in st.session_state and st.session_state.dataset_ready:
            X = playground.datasets['current']['X']
            y = playground.datasets['current']['y']
            feature_names = playground.datasets['current']['feature_names']
            
            # Data visualization
            st.subheader("Data Visualization")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['2D PCA Projection', 'Feature Distribution', 
                              'Correlation Heatmap', 'Class Distribution'],
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                      [{"type": "heatmap"}, {"type": "bar"}]]
            )
            
            # PCA Projection
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            fig.add_trace(
                go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers',
                          marker=dict(color=y, colorscale='viridis')),
                row=1, col=1
            )
            
            # Feature distribution
            feature_means = np.mean(X, axis=0)
            fig.add_trace(
                go.Histogram(x=feature_means, nbinsx=20),
                row=1, col=2
            )
            
            # Correlation heatmap
            corr_matrix = np.corrcoef(X.T)
            fig.add_trace(
                go.Heatmap(z=corr_matrix, colorscale='RdBu_r'),
                row=2, col=1
            )
            
            # Class distribution
            unique, counts = np.unique(y, return_counts=True)
            fig.add_trace(
                go.Bar(x=unique, y=counts),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Dataset statistics
            st.subheader("Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Samples", X.shape[0])
            with col2:
                st.metric("Features", X.shape[1])
            with col3:
                st.metric("Classes", len(np.unique(y)))
            with col4:
                st.metric("Missing Values", 0)

def model_trainer(playground):
    st.header("🎯 Interactive Model Trainer")

    if not st.session_state.get('dataset_ready', False):
        st.warning("Please generate a dataset in the Data Explorer first!")
        return
    
    X = playground.datasets['current']['X']
    y = playground.datasets['current']['y']
    feature_names = playground.datasets['current']['feature_names']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Select Algorithm",
            ["Random Forest", "Gradient Boosting", "SVM", "K-Means", 
             "DBSCAN", "Gaussian Mixture", "Isolation Forest"]
        )
        
        # Algorithm-specific parameters
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 500, 100)
            max_depth = st.slider("Max Depth", 2, 20, 10)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth}
            
        elif model_type == "Gradient Boosting":
            n_estimators = st.slider("Number of Trees", 10, 500, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
            params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
            
        elif model_type == "SVM":
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
            params = {'kernel': kernel, 'C': C}
            
        elif model_type == "K-Means":
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)
            params = {'n_clusters': n_clusters}
            
        elif model_type == "DBSCAN":
            eps = st.slider("EPS", 0.1, 2.0, 0.5)
            min_samples = st.slider("Min Samples", 2, 20, 5)
            params = {'eps': eps, 'min_samples': min_samples}
            
        elif model_type == "Gaussian Mixture":
            n_components = st.slider("Number of Components", 2, 10, 4)
            params = {'n_components': n_components}
            
        elif model_type == "Isolation Forest":
            n_estimators = st.slider("Number of Trees", 10, 500, 100)
            params = {'n_estimators': n_estimators}
        
        if st.button("Train Model", use_container_width=True):
            with st.spinner(f"Training {model_type}..."):
                model= playground.train_model(model_type, X, y, **params)
                # Split data for supervised learning
                if model_type in ["Random Forest", "Gradient Boosting", "SVM"]:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    model = playground.train_model(model_type, X_train, y_train, **params)
                    
                    # Make predictions
                    if model_type == "SVM":
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)
                    else:
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Store results
                    playground.results['current'] = {
                        'model': model,
                        'type': model_type,
                        'accuracy': accuracy,
                        'predictions': y_pred,
                        'probabilities': y_proba,
                        'test_set': (X_test, y_test)
                    }
                    
                else:  # Unsupervised learning
                    model = playground.train_model(model_type, X, y, **params)
                    
                    if model_type == "K-Means":
                        labels = model.labels_
                        silhouette = silhouette_score(X, labels)
                    elif model_type == "Gaussian Mixture":
                        labels = model.predict(X)
                        silhouette = silhouette_score(X, labels)
                    elif model_type == "DBSCAN":
                        labels = model.labels_
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        if n_clusters > 1:
                            silhouette = silhouette_score(X, labels)
                        else:
                            silhouette = -1
                    elif model_type == "Isolation Forest":
                        labels = model.predict(X)
                        labels = (labels == -1).astype(int)  # Convert to 0/1
                    
                    playground.results['current'] = {
                        'model': model,
                        'type': model_type,
                        'labels': labels,
                        'silhouette': silhouette if 'silhouette' in locals() else None
                    }
                
                st.session_state.model_trained = True
    
    with col2:
        if 'model_trained' in st.session_state and st.session_state.model_trained:
            results = playground.results['current']
            model_type = results['type']
            
            st.subheader(f"{model_type} Results")
            
            if model_type in ["Random Forest", "Gradient Boosting", "SVM"]:
                # Supervised learning results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.3f}")
                with col2:
                    cv_scores = cross_val_score(results['model'], X, y, cv=5)
                    st.metric("CV Score", f"{np.mean(cv_scores):.3f}")
                with col3:
                    st.metric("Model Type", "Supervised")
                
                # Confusion Matrix
                X_test, y_test = results['test_set']
                cm = confusion_matrix(y_test, results['predictions'])
                
                fig_cm = px.imshow(cm, text_auto=True, 
                                 title="Confusion Matrix",
                                 color_continuous_scale='Blues')
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Feature Importance (for tree-based models)
                if model_type in ["Random Forest", "Gradient Boosting"]:
                    st.subheader("Feature Importance")
                    importances = results['model'].feature_importances_
                    feature_imp_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=True)
                    
                    fig_imp = px.bar(feature_imp_df, x='importance', y='feature',
                                   orientation='h', title="Feature Importance",
                                   color='importance', color_continuous_scale='viridis')
                    st.plotly_chart(fig_imp, use_container_width=True)
            
            else:
                # Unsupervised learning results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    n_clusters = len(np.unique(results['labels']))
                    st.metric("Clusters Found", n_clusters)
                with col2:
                    if results['silhouette'] is not None and results['silhouette'] != -1:
                        st.metric("Silhouette Score", f"{results['silhouette']:.3f}")
                    else:
                        st.metric("Silhouette Score", "N/A")
                with col3:
                    st.metric("Model Type", "Unsupervised")
                
                # Cluster visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                fig_clusters = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                                        color=results['labels'].astype(str),
                                        title="Cluster Visualization (PCA)",
                                        labels={'color': 'Cluster'})
                st.plotly_chart(fig_clusters, use_container_width=True)

def feature_analysis(playground):
    st.header("🔍 Advanced Feature Analysis")
    
    if not st.session_state.get('dataset_ready', False):
        st.warning("Please generate a dataset in the Data Explorer first!")
        return
    
    X = playground.datasets['current']['X']
    y = playground.datasets['current']['y']
    feature_names = playground.datasets['current']['feature_names']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Analysis Methods")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Feature Importance", "Dimensionality Reduction", "Feature Selection"]
        )
        
        if analysis_type == "Dimensionality Reduction":
            dr_method = st.selectbox(
                "Select Method",
                ["PCA", "Kernel PCA", "t-SNE"]
            )
            
            n_components = st.slider("Number of Components", 2, 5, 2)
            
        elif analysis_type == "Feature Selection":
            fs_method = st.selectbox(
                "Select Method",
                ["K-Best", "Random Forest"]
            )
            n_features = st.slider("Number of Features to Select", 2, X.shape[1], 5)
    
    with col2:
        if analysis_type == "Feature Importance":
            st.subheader("Comparative Feature Importance")
            
            # Calculate importance using multiple methods
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
            
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb.fit(X, y)
            gb_importance = gb.feature_importances_
            
            # Create comparison plot
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Random Forest': rf_importance,
                'Gradient Boosting': gb_importance
            })
            
            fig = go.Figure()
            for method in ['Random Forest', 'Gradient Boosting']:
                fig.add_trace(go.Bar(
                    name=method,
                    x=importance_df['Feature'],
                    y=importance_df[method],
                    opacity=0.7
                ))
            
            fig.update_layout(barmode='group', title="Feature Importance Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Dimensionality Reduction":
            st.subheader(f"{dr_method} Visualization")
            
            if dr_method == "PCA":
                reducer = PCA(n_components=n_components)
            elif dr_method == "Kernel PCA":
                reducer = KernelPCA(n_components=n_components, kernel='rbf')
            else:  # t-SNE
                reducer = TSNE(n_components=n_components, random_state=42)
            
            X_reduced = reducer.fit_transform(X)
            
            if n_components == 2:
                fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=y,
                               title=f"{dr_method} Projection")
            else:
                fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                  color=y, title=f"{dr_method} Projection")
            
            st.plotly_chart(fig, use_container_width=True)
            
            if dr_method == "PCA":
                explained_variance = reducer.explained_variance_ratio_
                st.write("Explained Variance Ratio:", explained_variance)

def model_benchmark(playground):
    st.header("📈 Model Benchmarking Suite")
    
    if not st.session_state.get('dataset_ready', False):
        st.warning("Please generate a dataset in the Data Explorer first!")
        return
    
    X = playground.datasets['current']['X']
    y = playground.datasets['current']['y']
    
    st.subheader("Algorithm Performance Comparison")
    
    # Define models to benchmark
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    }
    
    # Perform benchmarking
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'CV Mean': np.mean(cv_scores),
            'CV Std': np.std(cv_scores),
            'Training Time': np.random.uniform(0.1, 2.0)  # Simulated timing
        })
    
    results_df = pd.DataFrame(results)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        
        # Create radar chart for model comparison
        metrics = ['Accuracy', 'CV Mean', 'Training Time']
        fig = go.Figure()
        
        for idx, row in results_df.iterrows():
            values = [row['Accuracy'], row['CV Mean'], 1/row['Training Time']]  # Inverse for better visualization
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                         showlegend=True,
                         title="Model Comparison Radar Chart")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Detailed Results")
        
        # Performance table
        st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'CV Mean']))
        
        # Bar chart comparison
        fig_bar = px.bar(results_df, x='Model', y=['Accuracy', 'CV Mean'],
                        barmode='group', title="Model Performance Comparison")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Best model recommendation
        best_model = results_df.loc[results_df['Accuracy'].idxmax()]
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏆 Best Performing Model</h3>
            <p><b>{best_model['Model']}</b> with {best_model['Accuracy']:.3f} accuracy</p>
            <p>Cross-validation: {best_model['CV Mean']:.3f} ± {best_model['CV Std']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()