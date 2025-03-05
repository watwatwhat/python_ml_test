import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# データセットの選択（load_iris, load_wine, load_breast_cancer から選択）
data_choice = 'breast_cancer'  # 'iris', 'wine', 'breast_cancer'

data_loaders = {
    'iris': load_iris,
    'wine': load_wine,
    'breast_cancer': load_breast_cancer
}

data = data_loaders[data_choice]()
X = data.data
y = data.target

# データの統計情報を表示
print("データの統計情報:")
print(pd.DataFrame(X, columns=data.feature_names).describe())

# 訓練データとテストデータに分割（80%:訓練, 20%:テスト）
test_size = 0.2  # 変更可能
random_state = 42  # 乱数シード固定
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# 訓練データ・テストデータの一部を表示
print("\n訓練データのサンプル:")
print(pd.DataFrame(X_train[:5], columns=data.feature_names))
print("\n対応するラベル:")
print(y_train[:5])

# モデルの作成と学習
n_estimators = 100  # 決定木の数（変更可）
model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度評価
accuracy = accuracy_score(y_test, y_pred)
print(f"\nモデルの精度: {accuracy:.2f}")

# 混同行列の表示
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n混同行列:")
print(conf_matrix)

# 詳細な分類レポート
print("\n分類レポート:")
print(classification_report(y_test, y_pred))

# 混同行列の可視化
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
