#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Load the new dataset
file_path = './Marketing.csv'
marketing_data = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# 1.1 Separate features and labels
X = marketing_data.drop(columns=['successful_marketing'])
y = marketing_data['successful_marketing']
# 1.2 Handling categorical variables using OneHotEncoder
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
# Define the preprocessing for both numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
# 1.3 Define the preprocessing pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', MinMaxScaler())
])
# Apply the preprocessing pipeline to the features
X_processed = pipeline.fit_transform(X)
#%%
# Step 3: Feature Extraction
# 3.1 Split data into training and test sets (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# 3.2 Apply PCA on the training data to create 3 principal components
pca = PCA(n_components=5, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_train_pca[:, 0], 
    y=X_train_pca[:, 1], 
    hue=y_train, 
    palette='Set1', 
    s=100
)
plt.title('PCA of Marketing Dataset (First Two Principal Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Outcome')
plt.show()
#%%
# Step 4: Classification using a Super Learner
# Define base classifiers
nb = GaussianNB()
nn = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
knn = KNeighborsClassifier()

# Define meta-learner (Decision Tree)
meta_learner = DecisionTreeClassifier(random_state=42)

# Define super learner using stacking
super_learner = StackingClassifier(
    estimators=[('nb', nb), ('nn', nn), ('knn', knn)],
    final_estimator=meta_learner,
    cv=5
)

# Hyperparameter tuning (expanded parameter grid)
param_grid = {
    'final_estimator__max_depth': [3, 5, 7, 9, 11],
    'final_estimator__min_samples_split': [2, 5, 10, 15, 20],
    'final_estimator__min_samples_leaf': [1, 2, 4],
    'final_estimator__criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(super_learner, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model
grid_search.fit(X_train_pca, y_train)

# Evaluate the model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_pca)
test_accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Best Hyperparameters:", grid_search.best_params_)
print("Test Accuracy:", test_accuracy)
# %%
