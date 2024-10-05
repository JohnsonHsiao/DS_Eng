#%%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# 读取训练和测试数据集
train_data = pd.read_csv('hw4_train.csv')
test_data = pd.read_csv('hw4_test.csv')

#%%
X_train = train_data.drop(columns=['BloodPressure']) 
y_train = train_data['BloodPressure'] 

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

X_test = test_data.drop(columns=['BloodPressure'])  
test_data['BloodPressure'] = reg_model.predict(X_test)

test_data.to_csv('test_data_with_predictions.csv', index=False)
print("test_data_with_predictions.csv is saved")

#%%
X_train_knn = train_data.drop(columns=['Outcome'])
y_train_knn = train_data['Outcome']
X_test_knn = test_data.drop(columns=['Outcome'])
y_test_knn = test_data['Outcome']

knn_accuracies = []

for k in range(1, 20):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn_model.predict(X_test_knn)
    
    accuracy = accuracy_score(y_test_knn, y_pred_knn)
    knn_accuracies.append((k, accuracy))

best_k, best_accuracy = max(knn_accuracies, key=lambda x: x[1])

print(f"The best K : {best_k}, accuracy : {best_accuracy:.4f}")

for k, accuracy in knn_accuracies:
    print(f"K={k}, accuracy={accuracy:.4f}")


# %%
