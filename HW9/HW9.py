# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the dataset
file_path = 'HW9/world_ds.csv'
data = pd.read_csv(file_path)

# Handle non-numeric values by converting categorical features to numeric using one-hot encoding
data = pd.get_dummies(data)

# Split features and label
X = data.drop(columns=['development_status'])
y = data['development_status']

# 1. Forward wrapper method to select best three features
# Use Logistic Regression as estimator for feature selection
lr = LogisticRegression(max_iter=1000)
sfs = SFS(lr, k_features=3, forward=True, floating=False, scoring='accuracy', cv=5)
sfs = sfs.fit(X, y)

# Get the selected feature names
selected_features = list(sfs.k_feature_names_)
print("Selected features from forward wrapper method:", selected_features)

# Create a new dataset with the selected features
X_forward = X[selected_features]

# 2. Apply PCA to create 3 new components from existing features
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create a DataFrame for PCA components
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])

# Explain each principal component based on correlations with original features
pca_components = pd.DataFrame(pca.components_, columns=X.columns, index=['PC1', 'PC2', 'PC3'])
print("\nPCA Components (Correlation with original features):\n", pca_components)

# 3. Apply LDA to create 2 new components from existing features
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

# Create a DataFrame for LDA components
X_lda_df = pd.DataFrame(X_lda, columns=['LDA1', 'LDA2'])

# 4. Compare the accuracy of KNN classifier on selected features (forward), PCA, and LDA
# Split data into training and testing sets
X_train_forward, X_test_forward, y_train, y_test = train_test_split(X_forward, y, test_size=0.25, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca_df, y, test_size=0.25, random_state=42)
X_train_lda, X_test_lda, _, _ = train_test_split(X_lda_df, y, test_size=0.25, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train and evaluate KNN on forward selected features
knn.fit(X_train_forward, y_train)
y_pred_forward = knn.predict(X_test_forward)
accuracy_forward = accuracy_score(y_test, y_pred_forward)
print("\nAccuracy of KNN on forward selected features:", accuracy_forward)

# Train and evaluate KNN on PCA components
knn.fit(X_train_pca, y_train)
y_pred_pca = knn.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("Accuracy of KNN on PCA components:", accuracy_pca)

# Train and evaluate KNN on LDA components
knn.fit(X_train_lda, y_train)
y_pred_lda = knn.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print("Accuracy of KNN on LDA components:", accuracy_lda)

# Summary of accuracies
print("\nSummary of KNN accuracies:")
print("Forward Selected Features Accuracy:", accuracy_forward)
print("PCA Components Accuracy:", accuracy_pca)
print("LDA Components Accuracy:", accuracy_lda)
