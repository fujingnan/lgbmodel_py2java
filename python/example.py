from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# # 加载数据
iris = load_iris()
data = iris.data
target = iris.target

# # 划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# # 模型训练
# gbm = LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
# gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
import pandas as pd
from lightgbm import LGBMClassifier
model = LGBMClassifier(boosting_type='gbdt', objective="multiclass", nthread=8, seed=42, num_leaves=31, learning_rate=0.05, n_estimators=20)
model.n_classes =3
model.fit(X_train, y_train)
model.booster_.save_model("lightgbm.txt")
