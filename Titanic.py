# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import preprocessing
import pickle
from matplotlib import style
import matplotlib.pyplot as pyplot

test_data = pd.read_csv("test.csv")
test_data.fillna(0,inplace=True)
train_data = pd.read_csv("train.csv")
train_data.fillna(0,inplace=True)
best = 0
features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
predict = "Survived"
data = train_data[features]
data.Sex[data.Sex == 'male'] = 1
data.Sex[data.Sex == 'female'] = 0
X = np.array(data)
y = np.array(train_data[predict])

best = 0

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


for _ in range(30):
   x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.15)
    
   model = linear_model.Ridge(alpha=.5)
    
   model.fit(x_train,y_train)
   if model.score(x_test, y_test) > best:
       best = model.score(x_test, y_test)
       with open("titanic.pickle","wb") as f:
           pickle.dump(model,f)
            
pickle_in = open("titanic.pickle","rb")
model = pickle.load(pickle_in)
print(model.score(x_test,y_test))


predicted = {}
test_data.Sex[test_data.Sex == 'male'] = 1
test_data.Sex[test_data.Sex == 'female'] = 0
predictions = model.predict(test_data[features])
for x in range(len(predictions)):
    predicted[x] = int(round(predictions[x]))
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predicted})
output.to_csv('my_submission.csv', index=False)

