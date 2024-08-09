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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, auc, accuracy_score)
import matplotlib.pyplot as plt

train = train.copy()
valid = valid.copy()

# 説明変数を抽出
X_train = train.drop('Survived', axis=1)
X_valid = valid.drop('Survived', axis=1)

# 目的変数を抽出
y_train = train.Survived
y_valid = valid.Survived

# 特徴量作成処理を定義
def preprocess(X):
    # 不要な列を削除
    X = X.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)

    # 欠損値処理
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Embarked'] = X['Embarked'].fillna('S')

    # カテゴリ変数の変換
    X['Sex'] = X['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    return X

# 特徴量作成
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)

# モデルにランダムフォレストを使用
clf = RandomForestClassifier(random_state=0, n_estimators=10)

# モデルを訓練
clf = clf.fit(X_train, y_train)

# 検証データで予測
pred = clf.predict(X_valid)

# 予測結果の評価指標を確認
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# 正解率 (Accuracy)
print("Accuracy : ", accuracy_score(y_valid, pred))
# 精度 (Precision)
print("Precision: ", precision_score(y_valid, pred))
# 検出率 (Recall)
print("Recall   : ", recall_score(y_valid, pred))
# F値 (F-measure) 
print("F-measure: ", f1_score(y_valid, pred))