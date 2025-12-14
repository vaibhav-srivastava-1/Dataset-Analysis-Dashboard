"""
Comprehensive Data Science Project
Demonstrating Supervised Learning Algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                            classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 60)
print("COMPREHENSIVE DATA SCIENCE PROJECT")
print("Supervised Learning Algorithms Demonstration")
print("=" * 60)

# ============================================================================
# PART 1: REGRESSION PROBLEMS
# ============================================================================

print("\n" + "=" * 60)
print("PART 1: REGRESSION ANALYSIS")
print("=" * 60)

# Generate synthetic regression dataset
np.random.seed(42)
n_samples = 500
X_reg = np.random.rand(n_samples, 3) * 10
# Create a non-linear relationship with some noise
y_reg = (2 * X_reg[:, 0] + 3 * X_reg[:, 1]**2 - 1.5 * X_reg[:, 2] + 
         np.random.normal(0, 2, n_samples))

# Create DataFrame for better visualization
df_reg = pd.DataFrame(X_reg, columns=['Feature_1', 'Feature_2', 'Feature_3'])
df_reg['Target'] = y_reg

print("\nRegression Dataset Info:")
print(df_reg.head())
print(f"\nDataset shape: {df_reg.shape}")
print(f"\nDataset Statistics:")
print(df_reg.describe())

# Visualize the regression data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Regression Dataset Visualization', fontsize=16)

# Correlation heatmap
sns.heatmap(df_reg.corr(), annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
axes[0, 0].set_title('Correlation Heatmap')

# Feature distributions
df_reg[['Feature_1', 'Feature_2', 'Feature_3']].boxplot(ax=axes[0, 1])
axes[0, 1].set_title('Feature Distributions')

# Target distribution
axes[1, 0].hist(df_reg['Target'], bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Target Distribution')
axes[1, 0].set_xlabel('Target Value')
axes[1, 0].set_ylabel('Frequency')

# Feature vs Target scatter
axes[1, 1].scatter(df_reg['Feature_1'], df_reg['Target'], alpha=0.5)
axes[1, 1].set_title('Feature_1 vs Target')
axes[1, 1].set_xlabel('Feature_1')
axes[1, 1].set_ylabel('Target')

plt.tight_layout()
plt.savefig('regression_data_visualization.png', dpi=300, bbox_inches='tight')
print("\nSaved: regression_data_visualization.png")

# Prepare data for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# Store regression results
regression_results = {}

# 1. Linear Regression
print("\n" + "-" * 60)
print("1. LINEAR REGRESSION")
print("-" * 60)
lr = LinearRegression()
lr.fit(X_reg_train_scaled, y_reg_train)
y_pred_lr = lr.predict(X_reg_test_scaled)
mse_lr = mean_squared_error(y_reg_test, y_pred_lr)
r2_lr = r2_score(y_reg_test, y_pred_lr)
regression_results['Linear Regression'] = {'MSE': mse_lr, 'R2': r2_lr}
print(f"MSE: {mse_lr:.4f}")
print(f"R2 Score: {r2_lr:.4f}")

# 2. Polynomial Regression
print("\n" + "-" * 60)
print("2. POLYNOMIAL REGRESSION (degree=2)")
print("-" * 60)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly_features.fit_transform(X_reg_train_scaled)
X_poly_test = poly_features.transform(X_reg_test_scaled)
poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_reg_train)
y_pred_poly = poly_reg.predict(X_poly_test)
mse_poly = mean_squared_error(y_reg_test, y_pred_poly)
r2_poly = r2_score(y_reg_test, y_pred_poly)
regression_results['Polynomial Regression'] = {'MSE': mse_poly, 'R2': r2_poly}
print(f"MSE: {mse_poly:.4f}")
print(f"R2 Score: {r2_poly:.4f}")

# 3. Ridge Regression
print("\n" + "-" * 60)
print("3. RIDGE REGRESSION (alpha=1.0)")
print("-" * 60)
ridge = Ridge(alpha=1.0)
ridge.fit(X_reg_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_reg_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
regression_results['Ridge Regression'] = {'MSE': mse_ridge, 'R2': r2_ridge}
print(f"MSE: {mse_ridge:.4f}")
print(f"R2 Score: {r2_ridge:.4f}")

# 4. Lasso Regression
print("\n" + "-" * 60)
print("4. LASSO REGRESSION (alpha=0.1)")
print("-" * 60)
lasso = Lasso(alpha=0.1)
lasso.fit(X_reg_train_scaled, y_reg_train)
y_pred_lasso = lasso.predict(X_reg_test_scaled)
mse_lasso = mean_squared_error(y_reg_test, y_pred_lasso)
r2_lasso = r2_score(y_reg_test, y_pred_lasso)
regression_results['Lasso Regression'] = {'MSE': mse_lasso, 'R2': r2_lasso}
print(f"MSE: {mse_lasso:.4f}")
print(f"R2 Score: {r2_lasso:.4f}")

# Visualize regression results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Regression Model Predictions', fontsize=16)

models = [
    ('Linear Regression', y_pred_lr),
    ('Polynomial Regression', y_pred_poly),
    ('Ridge Regression', y_pred_ridge),
    ('Lasso Regression', y_pred_lasso)
]

for idx, (name, y_pred) in enumerate(models):
    ax = axes[idx // 2, idx % 2]
    ax.scatter(y_reg_test, y_pred, alpha=0.6)
    ax.plot([y_reg_test.min(), y_reg_test.max()], 
            [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{name}\nR2: {regression_results[name]["R2"]:.4f}')

plt.tight_layout()
plt.savefig('regression_predictions.png', dpi=300, bbox_inches='tight')
print("\nSaved: regression_predictions.png")

# ============================================================================
# PART 2: CLASSIFICATION PROBLEMS
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: CLASSIFICATION ANALYSIS")
print("=" * 60)

# Generate synthetic classification dataset
np.random.seed(42)
n_samples = 1000
X_clf = np.random.randn(n_samples, 4)
# Create a non-linear decision boundary
y_clf = ((X_clf[:, 0]**2 + X_clf[:, 1]**2 + 
         0.5 * X_clf[:, 2] + 0.3 * X_clf[:, 3]) > 2).astype(int)

# Create DataFrame
df_clf = pd.DataFrame(X_clf, columns=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4'])
df_clf['Target'] = y_clf

print("\nClassification Dataset Info:")
print(df_clf.head())
print(f"\nDataset shape: {df_clf.shape}")
print(f"\nTarget Distribution:")
print(df_clf['Target'].value_counts())

# Visualize the classification data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Classification Dataset Visualization', fontsize=16)

# Correlation heatmap
sns.heatmap(df_clf.corr(), annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
axes[0, 0].set_title('Correlation Heatmap')

# Feature distributions by class
for i, feature in enumerate(['Feature_1', 'Feature_2']):
    ax = axes[0, 1] if i == 0 else axes[1, 0]
    df_clf.boxplot(column=feature, by='Target', ax=ax)
    ax.set_title(f'{feature} by Class')
    ax.set_xlabel('Class')

# Target distribution
df_clf['Target'].value_counts().plot(kind='bar', ax=axes[1, 1], color=['skyblue', 'salmon'])
axes[1, 1].set_title('Class Distribution')
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_xticklabels(['Class 0', 'Class 1'], rotation=0)

plt.tight_layout()
plt.savefig('classification_data_visualization.png', dpi=300, bbox_inches='tight')
print("\nSaved: classification_data_visualization.png")

# Prepare data for classification
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Scale features
scaler_clf = StandardScaler()
X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train)
X_clf_test_scaled = scaler_clf.transform(X_clf_test)

# Store classification results
classification_results = {}

# 5. Logistic Regression
print("\n" + "-" * 60)
print("5. LOGISTIC REGRESSION")
print("-" * 60)
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_clf_train_scaled, y_clf_train)
y_pred_log = log_reg.predict(X_clf_test_scaled)
acc_log = accuracy_score(y_clf_test, y_pred_log)
classification_results['Logistic Regression'] = {'Accuracy': acc_log}
print(f"Accuracy: {acc_log:.4f}")
print("\nClassification Report:")
print(classification_report(y_clf_test, y_pred_log))

# 6. K-Nearest Neighbors (KNN)
print("\n" + "-" * 60)
print("6. K-NEAREST NEIGHBORS (KNN, k=5)")
print("-" * 60)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_clf_train_scaled, y_clf_train)
y_pred_knn = knn.predict(X_clf_test_scaled)
acc_knn = accuracy_score(y_clf_test, y_pred_knn)
classification_results['KNN'] = {'Accuracy': acc_knn}
print(f"Accuracy: {acc_knn:.4f}")
print("\nClassification Report:")
print(classification_report(y_clf_test, y_pred_knn))

# 7. Support Vector Machine (SVM)
print("\n" + "-" * 60)
print("7. SUPPORT VECTOR MACHINE (SVM)")
print("-" * 60)
svm = SVC(kernel='rbf', random_state=42, probability=True)
svm.fit(X_clf_train_scaled, y_clf_train)
y_pred_svm = svm.predict(X_clf_test_scaled)
acc_svm = accuracy_score(y_clf_test, y_pred_svm)
classification_results['SVM'] = {'Accuracy': acc_svm}
print(f"Accuracy: {acc_svm:.4f}")
print("\nClassification Report:")
print(classification_report(y_clf_test, y_pred_svm))

# 8. Decision Tree
print("\n" + "-" * 60)
print("8. DECISION TREE")
print("-" * 60)
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_clf_train_scaled, y_clf_train)
y_pred_dt = dt.predict(X_clf_test_scaled)
acc_dt = accuracy_score(y_clf_test, y_pred_dt)
classification_results['Decision Tree'] = {'Accuracy': acc_dt}
print(f"Accuracy: {acc_dt:.4f}")
print("\nClassification Report:")
print(classification_report(y_clf_test, y_pred_dt))

# 9. Random Forest
print("\n" + "-" * 60)
print("9. RANDOM FOREST CLASSIFIER")
print("-" * 60)
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_clf_train_scaled, y_clf_train)
y_pred_rf = rf.predict(X_clf_test_scaled)
acc_rf = accuracy_score(y_clf_test, y_pred_rf)
classification_results['Random Forest'] = {'Accuracy': acc_rf}
print(f"Accuracy: {acc_rf:.4f}")
print("\nClassification Report:")
print(classification_report(y_clf_test, y_pred_rf))

# Visualize classification results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Classification Model Results', fontsize=16)

models_clf = [
    ('Logistic Regression', y_pred_log),
    ('KNN', y_pred_knn),
    ('SVM', y_pred_svm),
    ('Decision Tree', y_pred_dt),
    ('Random Forest', y_pred_rf)
]

for idx, (name, y_pred) in enumerate(models_clf):
    ax = axes[idx // 3, idx % 3]
    cm = confusion_matrix(y_clf_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name}\nAccuracy: {classification_results[name]["Accuracy"]:.4f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

# Remove the last subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
print("\nSaved: classification_results.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)

print("\nREGRESSION MODELS:")
print("-" * 60)
reg_df = pd.DataFrame(regression_results).T
print(reg_df.sort_values('R2', ascending=False))

print("\nCLASSIFICATION MODELS:")
print("-" * 60)
clf_df = pd.DataFrame(classification_results).T
print(clf_df.sort_values('Accuracy', ascending=False))

# Create comparison plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Regression comparison
reg_df_sorted = reg_df.sort_values('R2', ascending=True)
axes[0].barh(reg_df_sorted.index, reg_df_sorted['R2'], color='steelblue')
axes[0].set_xlabel('R2 Score')
axes[0].set_title('Regression Models - R2 Score Comparison')
axes[0].grid(axis='x', alpha=0.3)

# Classification comparison
clf_df_sorted = clf_df.sort_values('Accuracy', ascending=True)
axes[1].barh(clf_df_sorted.index, clf_df_sorted['Accuracy'], color='coral')
axes[1].set_xlabel('Accuracy')
axes[1].set_title('Classification Models - Accuracy Comparison')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: model_comparison.png")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nGenerated Files:")
print("1. regression_data_visualization.png")
print("2. regression_predictions.png")
print("3. classification_data_visualization.png")
print("4. classification_results.png")
print("5. model_comparison.png")
print("\n" + "=" * 60)

