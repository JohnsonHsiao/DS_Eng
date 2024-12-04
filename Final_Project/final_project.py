
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './diabetes_project.csv'
data = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# 1.1 Remove outliers: Consider values as outliers if they are beyond 1.5 times the interquartile range (IQR)
def remove_outliers(df):
    for column in df.select_dtypes(include='number').columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

data_cleaned = remove_outliers(data.copy())
data_cleaned.describe()
# 1.2 Impute missing values using SimpleImputer from the impute package
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data_cleaned), columns=data_cleaned.columns)

# 1.3 Normalize all columns: Scale data to the range [0, 1]
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)
data_normalized.isnull().sum()
#%%
# Step 2: Unsupervised Learning for Generating Labels
# 2.1 Use K-means clustering on Glucose, BMI, and Age to create two clusters
kmeans = KMeans(n_clusters=2, random_state=42)
data_normalized['Cluster'] = kmeans.fit_predict(data_normalized[['Glucose', 'BMI', 'Age']])

# 2.2 Assign 'Diabetes' to the cluster with higher average Glucose
cluster_averages = data_normalized.groupby('Cluster')['Glucose'].mean()
diabetes_cluster = cluster_averages.idxmax()
data_normalized['Outcome'] = data_normalized['Cluster'].apply(lambda x: 1 if x == diabetes_cluster else 0)

# Drop the intermediate 'Cluster' column
data_normalized.drop(columns=['Cluster'], inplace=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Glucose', 
    y='BMI', 
    hue='Outcome',  # Use 'Outcome' instead of 'Cluster'
    data=data_normalized, 
    palette='Set1',  # Use Set1 for distinct consistent colors for each cluster
    style='Outcome',  # Use 'Outcome' instead of 'Cluster'
    s=100
)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='X')
plt.title('K-means Clustering of Diabetes Dataset with 2 Clusters')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.legend()
plt.show()

#%%
# Step 3: Feature Extraction
# 3.1 Split data into training and test sets (20% for testing)
X = data_normalized.drop(columns=['Outcome'])
y = data_normalized['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.2 Apply PCA on the training data to create 3 principal components
pca = PCA(n_components=3, random_state=42)
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
plt.title('PCA of Diabetes Dataset (First Two Principal Components)')
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
conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix, display_labels=['No Diabetes', 'Diabetes']).plot(cmap='Blues')
plt.title('Confusion Matrix for Diabetes Classification')
plt.show()

# Analyzing the results
tn, fp, fn, tp = conf_matrix.ravel()

# Insights:
# - True Positives (tp): Cases where the model correctly predicted the presence of diabetes.
# - True Negatives (tn): Cases where the model correctly predicted the absence of diabetes.
# - False Positives (fp): Cases where the model incorrectly predicted diabetes (Type I error).
# - False Negatives (fn): Cases where the model missed diagnosing diabetes (Type II error).

print(f'True Positives (TP): {tp}')
print(f'True Negatives (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')

# Calculate additional metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')

# %%
