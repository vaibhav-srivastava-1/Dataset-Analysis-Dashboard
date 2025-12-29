from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             classification_report, confusion_matrix, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Data Science Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df

def detect_problem_type(target):
    unique_values = target.nunique()
    total_values = len(target)
    if pd.api.types.is_numeric_dtype(target):
        if unique_values > 10 and unique_values / total_values > 0.1:
            return "regression"
        elif unique_values <= 10:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def run_regression_analysis(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results = {}
    predictions = {}
    with st.spinner("Training Linear Regression..."):
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred = lr.predict(X_test_scaled)
        results['Linear Regression'] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        predictions['Linear Regression'] = (y_test, y_pred)
    with st.spinner("Training Polynomial Regression..."):
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly_train = poly_features.fit_transform(X_train_scaled)
        X_poly_test = poly_features.transform(X_test_scaled)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly_train, y_train)
        y_pred = poly_reg.predict(X_poly_test)
        results['Polynomial Regression'] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        predictions['Polynomial Regression'] = (y_test, y_pred)
    with st.spinner("Training Ridge Regression..."):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)
        results['Ridge Regression'] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        predictions['Ridge Regression'] = (y_test, y_pred)
    with st.spinner("Training Lasso Regression..."):
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train_scaled, y_train)
        y_pred = lasso.predict(X_test_scaled)
        results['Lasso Regression'] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        predictions['Lasso Regression'] = (y_test, y_pred)
    return results, predictions, X_test, y_test

def run_classification_analysis(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results = {}
    predictions = {}
    with st.spinner("Training Logistic Regression..."):
        log_reg = LogisticRegression(random_state=random_state, max_iter=1000)
        log_reg.fit(X_train_scaled, y_train)
        y_pred = log_reg.predict(X_test_scaled)
        results['Logistic Regression'] = {
            'Accuracy': accuracy_score(y_test, y_pred)
        }
        predictions['Logistic Regression'] = (y_test, y_pred)
    with st.spinner("Training KNN..."):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        results['KNN'] = {
            'Accuracy': accuracy_score(y_test, y_pred)
        }
        predictions['KNN'] = (y_test, y_pred)
    with st.spinner("Training SVM..."):
        svm = SVC(kernel='rbf', random_state=random_state, probability=True)
        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_test_scaled)
        results['SVM'] = {
            'Accuracy': accuracy_score(y_test, y_pred)
        }
        predictions['SVM'] = (y_test, y_pred)
    with st.spinner("Training Decision Tree..."):
        dt = DecisionTreeClassifier(random_state=random_state, max_depth=5)
        dt.fit(X_train_scaled, y_train)
        y_pred = dt.predict(X_test_scaled)
        results['Decision Tree'] = {
            'Accuracy': accuracy_score(y_test, y_pred)
        }
        predictions['Decision Tree'] = (y_test, y_pred)
    with st.spinner("Training Random Forest..."):
        rf = RandomForestClassifier(
            n_estimators=100, random_state=random_state, max_depth=5)
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)
        results['Random Forest'] = {
            'Accuracy': accuracy_score(y_test, y_pred)
        }
        predictions['Random Forest'] = (y_test, y_pred)
    return results, predictions, X_test, y_test

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #e6edf3;
}
.block-container {
    padding-top: 1.5rem;
}
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    color: #58a6ff;
    margin-bottom: 1rem;
}
.card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card h3 {
    color: #58a6ff;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.card p,
.card li {
    color: #c9d1d9;
    font-size: 1.05rem;
    line-height: 1.7;
}
hr {
    border: none;
    height: 1px;
    background-color: #30363d;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📊 Data Science Analysis Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card">
<h3>📖 About This Dashboard</h3>
<p>
This interactive data science dashboard allows you to upload datasets,
automatically detect regression or vs classification problems, explore data
visually, train multiple machine learning models, and compare results.
</p>
<p>
<strong>Key Features:</strong> automated preprocessing, interactive plots,
multiple ML algorithms, best model recommendation, and cleaned dataset download.
</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your dataset file here"
)

st.sidebar.header("⚙️ Configuration")
target_column = None
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", 0, 100, 42, 1)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        st.subheader("📋 Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True, height=300)
        with st.expander("📊 View Full Dataset Information"):
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            st.write("**Data Types:**")
            st.dataframe(df.dtypes.to_frame('Data Type'),
                         use_container_width=True)
        st.subheader("🎯 Select Target Column")
        target_column = st.selectbox(
            "Choose the target column for prediction:",
            options=df.columns.tolist(),
            index=len(df.columns) - 1
        )
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            if X.isnull().sum().sum() > 0:
                st.warning(
                    "⚠️ Missing values detected. Filling with median (numeric) or mode (categorical).")
                for col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        X[col].fillna(X[col].mode()[0] if len(
                            X[col].mode()) > 0 else 'Unknown', inplace=True)
            categorical_cols = X.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.info(
                    f"📝 Found {len(categorical_cols)} categorical columns. Encoding them...")
                X_encoded = pd.get_dummies(
                    X, columns=categorical_cols, drop_first=True)
            else:
                X_encoded = X
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), name=y.name)
                st.info("📝 Target column encoded for classification.")
            problem_type = detect_problem_type(y)
            st.success(f"🔍 Detected Problem Type: **{problem_type.upper()}**")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Target Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                if problem_type == "classification":
                    y.value_counts().plot(kind='bar', ax=ax, color=[
                        'skyblue', 'salmon', 'lightgreen', 'orange', 'purple'])
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Count')
                    ax.set_title('Class Distribution')
                else:
                    ax.hist(y, bins=30, edgecolor='black',
                            alpha=0.7, color='steelblue')
                    ax.set_xlabel('Target Value')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Target Distribution')
                st.pyplot(fig)
            with col2:
                st.subheader("📈 Dataset Statistics")
                st.dataframe(df.describe(), use_container_width=True)
            st.subheader("📊 Interactive Data Visualizations")
            plot_type = st.selectbox(
                "Select visualization type:",
                ["Correlation Heatmap",
                    "Box Plot (Outlier Detection)", "Scatter Plot", "Bar Chart", "Histogram"]
            )
            if plot_type == "Correlation Heatmap":
                if X_encoded.shape[1] <= 20:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    df_viz = X_encoded.copy()
                    df_viz['Target'] = y
                    sns.heatmap(df_viz.corr(), annot=True,
                                cmap='coolwarm', center=0, ax=ax, fmt='.2f')
                    ax.set_title('Correlation Heatmap')
                    st.pyplot(fig)
                else:
                    st.warning(
                        "⚠️ Correlation heatmap disabled for datasets with more than 20 features (performance).")
            elif plot_type == "Box Plot (Outlier Detection)":
                numeric_cols = X.select_dtypes(
                    include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox(
                        "Select column for box plot:", numeric_cols)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[selected_col].plot(kind='box', ax=ax)
                    ax.set_title(
                        f'Box Plot: {selected_col} (Outlier Detection)')
                    ax.set_ylabel('Value')
                    st.pyplot(fig)
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[selected_col] < Q1 - 1.5*IQR)
                                  | (df[selected_col] > Q3 + 1.5*IQR)]
                    if len(outliers) > 0:
                        st.info(
                            f"📊 Found {len(outliers)} potential outliers (using IQR method)")
                else:
                    st.warning("⚠️ No numeric columns available for box plot.")
            elif plot_type == "Scatter Plot":
                numeric_cols = X.select_dtypes(
                    include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis:", numeric_cols)
                    with col2:
                        y_col = st.selectbox(
                            "Y-axis:", numeric_cols, index=min(1, len(numeric_cols)-1))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.warning(
                        "⚠️ Need at least 2 numeric columns for scatter plot.")
            elif plot_type == "Bar Chart":
                categorical_cols = X.select_dtypes(
                    include=['object']).columns.tolist()
                if len(categorical_cols) > 0:
                    selected_col = st.selectbox(
                        "Select categorical column:", categorical_cols)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts = df[selected_col].value_counts().head(20)
                    value_counts.plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title(f'Bar Chart: {selected_col}')
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning(
                        "⚠️ No categorical columns available for bar chart.")
            elif plot_type == "Histogram":
                numeric_cols = X.select_dtypes(
                    include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox(
                        "Select column for histogram:", numeric_cols)
                    bins = st.slider("Number of bins:", 10, 100, 30)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df[selected_col].dropna(), bins=bins,
                            edgecolor='black', alpha=0.7, color='steelblue')
                    ax.set_title(f'Histogram: {selected_col}')
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                else:
                    st.warning(
                        "⚠️ No numeric columns available for histogram.")
            st.markdown("---")
            st.subheader("💾 Download Cleaned Dataset")
            df_cleaned = df.copy()
            for col in df_cleaned.columns:
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col].fillna(
                        df_cleaned[col].median(), inplace=True)
                else:
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0] if len(
                        df_cleaned[col].mode()) > 0 else 'Unknown', inplace=True)
            csv = df_cleaned.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name=f"cleaned_{uploaded_file.name}",
                mime="text/csv",
                use_container_width=True,
                help="Download the dataset with missing values filled and ready for analysis"
            )
            st.info(
                "✅ The cleaned dataset has all missing values filled and is ready for further analysis!")
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                st.markdown("---")
                st.header("🤖 Model Training & Results")
                if problem_type == "regression":
                    results, predictions, X_test, y_test = run_regression_analysis(
                        X_encoded.values, y.values, test_size, random_state
                    )
                    st.subheader("📊 Regression Results")
                    results_df = pd.DataFrame(results).T
                    results_df = results_df.sort_values('R2', ascending=False)
                    st.dataframe(results_df.style.highlight_max(
                        axis=0, subset=['R2']), use_container_width=True)
                    st.subheader("📈 Model Predictions Visualization")
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    fig.suptitle('Regression Model Predictions', fontsize=16)
                    model_names = list(predictions.keys())
                    for idx, name in enumerate(model_names):
                        ax = axes[idx // 2, idx % 2]
                        y_test_vals, y_pred_vals = predictions[name]
                        ax.scatter(y_test_vals, y_pred_vals, alpha=0.6)
                        ax.plot([y_test_vals.min(), y_test_vals.max()],
                                [y_test_vals.min(), y_test_vals.max()], 'r--', lw=2)
                        ax.set_xlabel('Actual')
                        ax.set_ylabel('Predicted')
                        ax.set_title(f'{name}\nR2: {results[name]["R2"]:.4f}')
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.subheader("📊 Model Comparison")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    results_df_sorted = results_df.sort_values(
                        'R2', ascending=True)
                    ax.barh(results_df_sorted.index,
                            results_df_sorted['R2'], color='steelblue')
                    ax.set_xlabel('R2 Score')
                    ax.set_title('Regression Models - R2 Score Comparison')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
                    st.markdown("---")
                    st.subheader("🏆 Best Model Recommendation")
                    best_model = results_df.index[0]
                    best_metrics = results_df.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Model", best_model)
                    with col2:
                        st.metric("R² Score", f"{best_metrics['R2']:.4f}")
                    with col3:
                        st.metric("MSE", f"{best_metrics['MSE']:.4f}")
                    st.info(f"""
                    **🎯 Recommendation: Use {best_model}**
                    
                    This model performs best on your dataset with:
                    - **R² Score**: {best_metrics['R2']:.4f} (closer to 1.0 is better)
                    - **MSE**: {best_metrics['MSE']:.4f} (lower is better)
                    - **MAE**: {best_metrics['MAE']:.4f} (lower is better)
                    
                    **Why this model?**
                    - Highest R² score indicates best fit to your data
                    - Lower error metrics mean more accurate predictions
                    - Best balance between model complexity and performance
                    """)
                    with st.expander("📊 Detailed Model Insights"):
                        st.write("**Performance Ranking:**")
                        for idx, (model, row) in enumerate(results_df.iterrows(), 1):
                            if model == best_model:
                                st.write(
                                    f"🥇 **{idx}. {model}** (Best) - R²: {row['R2']:.4f}, MSE: {row['MSE']:.4f}")
                            else:
                                st.write(
                                    f"{idx}. {model} - R²: {row['R2']:.4f}, MSE: {row['MSE']:.4f}")
                        st.write("\n**Model Characteristics:**")
                        if "Polynomial" in best_model:
                            st.write(
                                "- Polynomial Regression captures non-linear relationships well")
                            st.write(
                                "- Good for complex datasets with curved patterns")
                        elif "Ridge" in best_model:
                            st.write(
                                "- Ridge Regression handles multicollinearity effectively")
                            st.write(
                                "- Prevents overfitting with L2 regularization")
                        elif "Lasso" in best_model:
                            st.write(
                                "- Lasso Regression performs feature selection automatically")
                            st.write("- Useful when you have many features")
                        else:
                            st.write(
                                "- Linear Regression provides simple, interpretable results")
                            st.write(
                                "- Best for linear relationships in your data")
                else:
                    results, predictions, X_test, y_test = run_classification_analysis(
                        X_encoded.values, y.values, test_size, random_state
                    )
                    st.subheader("📊 Classification Results")
                    results_df = pd.DataFrame(results).T
                    results_df = results_df.sort_values(
                        'Accuracy', ascending=False)
                    st.dataframe(results_df.style.highlight_max(
                        axis=0, subset=['Accuracy']), use_container_width=True)
                    st.subheader("📈 Confusion Matrices")
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle('Classification Model Results', fontsize=16)
                    model_names = list(predictions.keys())
                    for idx, name in enumerate(model_names):
                        ax = axes[idx // 3, idx % 3]
                        y_test_vals, y_pred_vals = predictions[name]
                        cm = confusion_matrix(y_test_vals, y_pred_vals)
                        sns.heatmap(cm, annot=True, fmt='d',
                                    cmap='Blues', ax=ax)
                        ax.set_title(
                            f'{name}\nAccuracy: {results[name]["Accuracy"]:.4f}')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                    axes[1, 2].axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.subheader("📊 Model Comparison")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    results_df_sorted = results_df.sort_values(
                        'Accuracy', ascending=True)
                    ax.barh(results_df_sorted.index,
                            results_df_sorted['Accuracy'], color='coral')
                    ax.set_xlabel('Accuracy')
                    ax.set_title('Classification Models - Accuracy Comparison')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
                    st.markdown("---")
                    st.subheader("🏆 Best Model Recommendation")
                    best_model = results_df.index[0]
                    best_accuracy = results_df.iloc[0]['Accuracy']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Best Model", best_model)
                    with col2:
                        st.metric(
                            "Accuracy", f"{best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
                    y_test_best, y_pred_best = predictions[best_model]
                    cm_best = confusion_matrix(y_test_best, y_pred_best)
                    report_best = classification_report(
                        y_test_best, y_pred_best, output_dict=True)
                    st.info(f"""
                    **🎯 Recommendation: Use {best_model}**
                    
                    This model performs best on your dataset with:
                    - **Accuracy**: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)
                    - Best overall performance across all classes
                    
                    **Why this model?**
                    - Highest accuracy means most correct predictions
                    - Best generalization to new data
                    - Optimal balance between precision and recall
                    """)
                    with st.expander("📊 Detailed Model Insights"):
                        st.write("**Performance Ranking:**")
                        for idx, (model, row) in enumerate(results_df.iterrows(), 1):
                            if model == best_model:
                                st.write(
                                    f"🥇 **{idx}. {model}** (Best) - Accuracy: {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")
                            else:
                                st.write(
                                    f"{idx}. {model} - Accuracy: {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")
                        st.write("\n**Best Model Performance Details:**")
                        if len(report_best) > 3:
                            for class_name, metrics in report_best.items():
                                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                    st.write(
                                        f"- **Class {class_name}**: Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
                        st.write("\n**Model Characteristics:**")
                        if "Random Forest" in best_model:
                            st.write(
                                "- Random Forest is robust and handles non-linear patterns well")
                            st.write(
                                "- Less prone to overfitting, good for complex datasets")
                        elif "SVM" in best_model:
                            st.write("- SVM finds optimal decision boundaries")
                            st.write("- Excellent for high-dimensional data")
                        elif "Decision Tree" in best_model:
                            st.write(
                                "- Decision Tree is interpretable and handles non-linear relationships")
                            st.write(
                                "- Good for understanding feature importance")
                        elif "KNN" in best_model:
                            st.write(
                                "- KNN works well with local patterns in data")
                            st.write(
                                "- Simple and effective for similar data points")
                        else:
                            st.write(
                                "- Logistic Regression provides interpretable results")
                            st.write("- Best for linear decision boundaries")
                    st.subheader("📋 Detailed Classification Reports")
                    for name in model_names:
                        with st.expander(f"View {name} Report"):
                            y_test_vals, y_pred_vals = predictions[name]
                            st.text(classification_report(
                                y_test_vals, y_pred_vals))
                st.success("✅ Analysis completed successfully!")
                st.balloons()
else:
    st.markdown("""
    <div class="card">
    <h3>👋 Welcome</h3>
    <p>
    Upload a CSV or Excel dataset from the sidebar to begin analysis.
    This dashboard will automatically detect whether your problem is
    regression or classification and run multiple machine learning models.
    </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h3>📖 How to Use</h3>
    <ol>
        <li>Upload a dataset (CSV / Excel)</li>
        <li>Select the target column</li>
        <li>Adjust test size and random state (optional)</li>
        <li>Explore visualizations</li>
        <li>Click <strong>Run Analysis</strong></li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
        <h3>📊 Regression Algorithms</h3>
        <ul>
            <li>Linear Regression</li>
            <li>Polynomial Regression</li>
            <li>Ridge Regression</li>
            <li>Lasso Regression</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
        <h3>🎯 Classification Algorithms</h3>
        <ul>
            <li>Logistic Regression</li>
            <li>K-Nearest Neighbors (KNN)</li>
            <li>Support Vector Machine (SVM)</li>
            <li>Decision Tree</li>
            <li>Random Forest</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h3>✨ Key Features</h3>
    <ul>
        <li>Automatic problem type detection</li>
        <li>Missing value handling</li>
        <li>Categorical feature encoding</li>
        <li>Interactive visualizations</li>
        <li>Multiple ML models comparison</li>
        <li>Best model recommendation</li>
        <li>Download cleaned dataset</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
