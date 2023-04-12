import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

# データセットのロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# データをフラットなベクトルに変換
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# SVMモデルの訓練
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

# テストデータに対する予測
y_pred = model.predict(x_test)

# 精度の計算
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

