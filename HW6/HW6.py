import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

data = pd.read_csv('diabetes.csv')

X = data.drop(columns=['Outcome'])  # Features (all columns except 'Outcome')
y = data['Outcome']  # Label (Outcome column)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Models with 3 and 50 estimators
models = {
    'RF_3': RandomForestClassifier(n_estimators=3, random_state=42),
    'RF_50': RandomForestClassifier(n_estimators=50, random_state=42),
    'AB_3': AdaBoostClassifier(n_estimators=3, random_state=42, algorithm='SAMME'),
    'AB_50': AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME')
}

results = {}

for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    results[model_name] = scores
    print(f"{model_name} Mean accuracy: {np.mean(scores):.4f} | Std: {np.std(scores):.4f}")

# Compare the mean of scores for RF and Adaboost with the same estimators
rf3_mean = np.mean(results['RF_3'])
ab3_mean = np.mean(results['AB_3'])
rf50_mean = np.mean(results['RF_50'])
ab50_mean = np.mean(results['AB_50'])

print("Comparison of models with the same number of estimators:")
print(f"RF_3 vs AB_3: {rf3_mean:.4f} vs {ab3_mean:.4f}")
print(f"RF_50 vs AB_50: {rf50_mean:.4f} vs {ab50_mean:.4f}")
