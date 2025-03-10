# 乳がん診断予測システム

このプロジェクトは、Pythonと機械学習の入門キットです。
01では単純なPythonスクリプトの実行を行い、02では機械学習モデルによる推論を行います。これを発展させ、03では機械学習モデルをWebアプリケーションとして構築し、外部からのリクエストに基づいて推論を行えるようにします。

## 機械学習シナリオ

機械学習を使用して乳がんの診断予測を行うシステムです。
Scikit-learnの乳がんデータセットを使用し、RandomForestClassifierでモデルを構築しています。

## プロジェクト構成

```
.
├── 01_get_datetime/          # 日時取得モジュール
├── 02_ml_cancer/            # 機械学習モデル
└── 03_ml_cancer_web/        # Webアプリケーション
```

## 環境構築

### 1. 機械学習モデル (02_ml_cancer)

```bash
cd 02_ml_cancer
pip install -r requirements.txt
```

必要なライブラリ:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

### 2. Webアプリケーション (03_ml_cancer_web)

```bash
cd 03_ml_cancer_web
pip install -r requirements.txt
```

必要なライブラリ:
- flask
- flask-cors
- numpy
- pandas
- scikit-learn
- joblib

## 実行手順

### 1. Webアプリケーションの起動

```bash
cd 03_ml_cancer_web
python app.py
```

アプリケーションは http://127.0.0.1:5000 で起動します。

### 2. 予測の実行

1. Webブラウザで http://127.0.0.1:5000 にアクセス
2. フォームに30個の特徴量を入力
   - サンプルデータを使用する場合は「サンプルデータを入力」ボタンをクリック
3. 「予測する」ボタンをクリックして結果を確認

## テストデータ

サンプルデータが用意されています。以下の手順でテストできます：

1. Webインターフェースの「サンプルデータを入力」ボタンをクリック
2. 自動的に全ての入力フィールドにサンプルデータが入力されます
3. 「予測する」ボタンをクリックして結果を確認

また、APIを直接テストする場合は、例えば以下のようなHTTPリクエストを使用できます：

```http
POST http://127.0.0.1:5000/predict
Content-Type: application/json

{
  "features": [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
  ]
}
```

`test_inference.http`から、テストリクエストを発行してみましょう。

## 予測結果

予測結果は以下のいずれかが返されます：
- 良性 (1)
- 悪性 (0)

エラーが発生した場合は、エラーメッセージが表示されます。
