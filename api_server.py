import pickle
import pandas as pd
from flask import Flask, request, jsonify

import sys
parent_dir = './pipelines'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from preprocessor import PreProcessor

# ファイルに保存した学習済みパイプラインをロードする
filename = './pipelines/ml_pipeline.pickle'
ml_pipeline = pickle.load(open(filename, 'rb'))

# Flaskサーバを作成
app = Flask(__name__)

# エンドポイントを定義 http:/host:port/predict, POSTメソッドのみ受け付ける
@app.route('/predict', methods=['POST'])
def post_predict():
    # Jsonリクエストから値取得
    X_dict = request.json

    # DataFrame化
    X_df = pd.DataFrame([X_dict])

    # パイプラインで予測を実行
    pred = ml_pipeline.predict(X_df)

    # 予測結果をJSON化
    result = {"Survived": int(pred[0])}

    # 予測結果を返信
    return jsonify(result)

if __name__ == '__main__':
    # Flaskサーバをポート番号5060で起動
    app.run(port=5060)