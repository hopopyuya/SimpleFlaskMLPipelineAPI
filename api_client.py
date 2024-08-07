import json
import urllib.request

url = 'http://127.0.0.1:5060/predict'
headers = {'Content-Type': 'application/json'}

if __name__ == '__main__':

    # 入力データを作成
    data = {
        'PassengerId': 294,
        'Pclass': 1,
        'Name': 'John, Mr. Smith',
        'Sex': 'male',
        'Age': 28,
        'SibSp': 0,
        'Parch': 0,
        'Ticket': 312453,
        'Fare': 18.2500,
        'Cabin': 'NaN',
        'Embarked': 'Q'
    }

    # 予測APIを呼び出し
    req = urllib.request.Request(url, json.dumps(data).encode(), headers)

    # 予測結果を確認
    with urllib.request.urlopen(req) as res:
        body = res.read()
        result = json.loads(body)
        print(result)