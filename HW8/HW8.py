import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

file_path = 'HW8/non_linear.csv'
data = pd.read_csv(file_path)

# Prepare the features and label
X = data.drop(columns=['label'])
y = data['label']

# Split the data into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train SVM with 'linear' kernel
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)

# Train SVM with 'rbf' kernel
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)

# Calculate accuracy for both models
y_pred_linear = svm_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)

y_pred_rbf = svm_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

# Report the best accuracy
best_accuracy = max(accuracy_linear, accuracy_rbf)

print('accuracy_linear : ', accuracy_linear)
print('accuracy_rbf : ', accuracy_rbf)
print('best_accuracy : ', best_accuracy)
