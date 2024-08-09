from sklearn.base import TransformerMixin

# 特徴量作成処理をTransformer化
class PreProcessor(TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        # 不要な列を削除
        X = X.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)

        #欠損値処理
        X['Fare'] = X['Fare'].fillna(X['Fare'].median())
        X['Age'] = X['Age'].fillna(X['Age'].median())
        X['Embarked'] = X['Embarked'].fillna('S')

        #カテゴリ変数の変換
        X['Sex'] = X['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        X['Embarked'] = X['Embarked'].map(
                {'S': 0, 'C': 1, 'Q': 2}).astype(int)

        return X