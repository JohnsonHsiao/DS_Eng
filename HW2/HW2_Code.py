import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import *

##########################################
##D-Tree Hyper-parameters
##########################################
max_depth_values = [3, 5]
min_samples_split_values = [5, 10]
min_samples_leaf_values = [3, 5]
min_impurity_decrease_values = [0.01, 0.001]
ccp_alpha_values = [0.001, 0.0001]

#Input Dateset
diabetes_df = pd.read_csv("diabetes.csv")

#Define features to predict Resistance label
features = diabetes_df.drop(columns='Outcome')
labels = diabetes_df['Outcome']

# Split data into train (72%), test (20%), and validation (8%)
train_feat, temp_feat, train_label, temp_label = train_test_split(features, labels, test_size=0.28, random_state=42)
test_feat, val_feat, test_label, val_label = train_test_split(temp_feat, temp_label, test_size=0.2857, random_state=42) # 0.2857 ≈ 20/70

#Create a model using Hyper-parameters
# To store the best parameters and highest accuracy
best_accuracy = 0
best_params = {}

# Loop through all combinations of hyperparameters using tqdm for progress visualization
for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        for min_samples_leaf in min_samples_leaf_values:
            for min_impurity_decrease in min_impurity_decrease_values:
                for ccp_alpha in ccp_alpha_values:
                    dtree = tree.DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_impurity_decrease=min_impurity_decrease,
                        ccp_alpha=ccp_alpha,
                        random_state=42
                    )
                    dtree.fit(train_feat, train_label)
                    val_score = dtree.score(val_feat, val_label)
                    if val_score > best_accuracy:
                        best_accuracy = val_score
                        best_params = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'min_impurity_decrease': min_impurity_decrease,
                            'ccp_alpha': ccp_alpha
                        }

print(best_params)
# Train the best model on the full training data and test it
best_model = tree.DecisionTreeClassifier(**best_params)
best_model.fit(train_feat, train_label)

# Predict on the test data
test_pred_label = best_model.predict(test_feat)
test_accuracy = accuracy_score(test_label, test_pred_label)

# Extracting the decision rules
rules = tree.export_text(best_model, feature_names=list(features.columns))

# Displaying test accuracy and decision rules
test_accuracy, rules.splitlines()[:10]  # Display the first few rules

plt.figure(figsize=(10, 10))
tree.plot_tree(best_model, feature_names=features.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Accuracy on training and test data
train_pred_label = best_model.predict(train_feat)
train_accuracy = accuracy_score(train_label, train_pred_label)

print(f"Training accuracy = {train_accuracy:.2f}")
print(f"Testing accuracy = {test_accuracy:.2f}")

# Confusion matrix for test data
conf_matrix = confusion_matrix(test_label, test_pred_label)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Diabetes', 'Diabetes']).plot()
plt.title("Confusion Matrix")
plt.show()

# Calculate accuracy, sensitivity, and specificity
accuracy = accuracy_score(test_label, test_pred_label)
sensitivity = recall_score(test_label, test_pred_label)
specificity = recall_score(test_label, test_pred_label, pos_label=0)

print(f"Accuracy: {accuracy:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")


# 1. If BMI ≤ 29.35 and Age ≤ 49.5, then No Diabetes
#    The samples that satisfy this condition have a Gini index of 0.077, with 25 samples classified as No Diabetes (value = [24, 1]).

# 2. If BMI > 29.35 and Glucose ≤ 146.5, then No Diabetes
#    The samples that satisfy this condition have a Gini index of 0.437, with 59 samples and a majority classified as Diabetes (value = [19, 40]).

# 3. If BMI > 29.35, Glucose > 146.5, and BMI ≤ 31.35, then Diabetes
#    The samples that satisfy this condition have a Gini index of 0.495, with 40 samples split between No Diabetes and Diabetes (value = [18, 22]).