import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# 1. 加载数据，根据实际数据集路径调整
df = pd.read_csv("data/train.csv")

# 2. 处理特征和标签
# 去除id列，以其余特征作为X，stroke列为标签y
X = df.drop(columns=["id", "stroke"])
y = df["stroke"]

# 处理缺失值，这里简单填充N/A为0，可根据实际情况优化
X = X.fillna(0)

# 对类别特征进行独热编码
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
X = pd.get_dummies(X, columns=categorical_cols)

# 3. 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 验证模型
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# 6. 保存模型
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/stroke_model.pkl")
print("模型已保存至 model/stroke_model.pkl")