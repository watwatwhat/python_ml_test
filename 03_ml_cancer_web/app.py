from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

# データセットのロード
data = load_breast_cancer()
X = data.data
y = data.target

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの作成と学習
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# モデルを保存
joblib.dump(model, 'model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSONデータを受け取る
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)  # 2D配列に変換
        
        # モデルをロード
        model = joblib.load('model.pkl')
        prediction = model.predict(features)
        
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
