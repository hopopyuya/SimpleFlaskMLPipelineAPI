from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

# 自作の特徴量作成処理をimport
from preprocessor import PreProcessor

import pandas as pd
from sklearn.model_selection import train_test_split

# データを読み込み
df = pd.read_csv('../csv/train.csv')

# データセットを、訓練用データと検証・評価用データに分割
(train, valid_test) = train_test_split(df, test_size = 0.3)

# 検証・評価用データを、検証用データと評価用データに分割
(valid, test) = train_test_split(valid_test, test_size = 0.5)

# 件数を確認
print("train: " + str(len(train)))
print("valid: " + str(len(valid)))
print("test : " + str(len(test)))

# 学習/予測パイプラインを定義
ml_pipeline = Pipeline([
        # 自作の特徴量作成を指定
        ('preprocessor', PreProcessor()), 
        # モデル（ランダムフォレスト）とハイパパラメータを指定
        ('random_forest', RandomForestClassifier(
                random_state=0, n_estimators=10))])

# 訓練データと評価データを用意
train = train.copy()
test = test.copy()

# 説明変数と目的を抽出
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)
y_train = train.Survived
y_test = test.Survived

# パイプラインで訓練を実行
ml_pipeline.fit(X_train, y_train)

# パイプラインで予測を実行
pred = ml_pipeline.predict(X_test)

# 予測結果の評価指標を確認
# 正解率 (Accuracy)
print("Accuracy : ", accuracy_score(y_test, pred))
# 精度 (Precision)
print("Precision: ", precision_score(y_test, pred))
# 検出率 (Recall)
print("Recall   : ", recall_score(y_test, pred))
# F値 (F-measure) 
print("F-measure: ", f1_score(y_test, pred))


import pickle

# パイプラインをファイルに保存する
filename = 'ml_pipeline.pickle'
pickle.dump(ml_pipeline, open(filename, 'wb'))