# 导入必要库
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 读取训练和测试数据集
train_data = pd.read_csv('hw4_train.csv')
test_data = pd.read_csv('hw4_test.csv')

# Step 1: 多重线性回归预测 BloodPressure
# 准备训练数据
X_train = train_data.drop(columns=['BloodPressure', 'Outcome'])  # 特征为所有非目标列
y_train = train_data['BloodPressure']  # 目标变量是 BloodPressure

# 初始化并训练回归模型
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# 在测试数据集上预测 BloodPressure
X_test = test_data.drop(columns=['BloodPressure', 'Outcome'])  # 测试集同样排除 BloodPressure 和 Outcome
test_data['BloodPressure'] = reg_model.predict(X_test)

# Step 2: KNN模型训练与测试
# 准备KNN模型的训练和测试数据 (使用预测的 BloodPressure 值)
X_train_knn = train_data.drop(columns=['Outcome'])
y_train_knn = train_data['Outcome']
X_test_knn = test_data.drop(columns=['Outcome'])
y_test_knn = test_data['Outcome']

# 存储不同 k 值的准确率
knn_accuracies = []

# 训练k从1到19的KNN模型并计算准确率
for k in range(1, 20):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn_model.predict(X_test_knn)
    
    # 计算准确率
    accuracy = accuracy_score(y_test_knn, y_pred_knn)
    knn_accuracies.append((k, accuracy))

# 找出准确率最高的K值
best_k, best_accuracy = max(knn_accuracies, key=lambda x: x[1])

# 输出最佳K值及其对应的准确率
print(f"最佳K值为: {best_k}, 对应的准确率为: {best_accuracy:.4f}")

# 输出每个K值对应的准确率
for k, accuracy in knn_accuracies:
    print(f"K={k}, 准确率={accuracy:.4f}")
