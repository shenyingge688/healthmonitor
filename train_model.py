# train_model.py
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. 创建存放模型的文件夹
if not os.path.exists('models'):
    os.makedirs('models')
    print("📁 已创建 'models' 文件夹")

print("⚙️ 正在生成模拟训练数据 (5000条)...")

# 2. 模拟生成 5000 条训练数据
X = [] # 存放特征
y = [] # 存放标签

for class_label in range(5):
    for _ in range(1000):
        # 生成 12 维特征
        features = [
            np.random.normal(75, 10),    # HR心率
            np.random.normal(50, 15),    # SDNN皮肤电反应次数
            np.random.normal(30, 10),    # RMSSD
            np.random.normal(20, 5),     # pNN50
            np.random.normal(1.2, 0.3),  # LF/HF
            np.random.normal(75, 10),    # Pulse Rate
            np.random.normal(0.5, 0.1),  # Amplitude
            np.random.normal(0, 0.5),    # Skewness
            np.random.normal(3, 1),      # Kurtosis
            np.random.normal(5, 2),      # EDA Level
            np.random.normal(0.2, 0.05), # EDA STD
            np.random.randint(1, 10)     # SCR Num
        ]
        
        # 加上一点疾病的明显特征，方便 AI 学习
        if class_label == 1: # 心律失常
            features[0] = np.random.normal(140, 15) 
        elif class_label == 2: # 睡眠呼吸暂停
            features[0] = np.random.normal(50, 8)
        elif class_label == 3: # 压力过载
            features[0] = np.random.normal(105, 10)
            features[10] = np.random.normal(0.4, 0.1)
        elif class_label == 4: # 自主神经紊乱
            features[0] = np.random.normal(105, 10)
            features[10] = np.random.normal(0.15, 0.05)
            
        X.append(features)
        y.append(class_label)

X = np.array(X)
y = np.array(y)

# 3. 划分数据并标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 训练随机森林模型
print("🧠 开始训练随机森林模型...")
rf_model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 5. 考试与评估
print("📊 正在进行模型评估...")
y_pred = rf_model.predict(X_test_scaled)
print(f"\n✅ 模型训练完成！准确率: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. 保存大脑文件
joblib.dump(rf_model, os.path.join('models', 'rf_model_5class.pkl'))
joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
print("💾 AI 大脑文件已成功保存至 'models' 文件夹！")