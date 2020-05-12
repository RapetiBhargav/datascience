import pandas as pd
import numpy as np
import os
from sklearn import tree, model_selection, preprocessing
import io
import pydot
import sklearn
print(sklearn.__version__)

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
print(titanic_train.columns)

lencoder = preprocessing.LabelEncoder()
lencoder.fit(titanic_train['Sex'])
print(lencoder.classes_)
titanic_train['Sex_encoded'] = lencoder.transform(titanic_train['Sex'])

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
titanic_train['Age_imputed'] =imp.fit_transform(titanic_train[['Age']]) 
print(imp.statistics_)

features = ['SibSp', 'Parch', 'Pclass', 'Sex_encoded', 'Age_imputed']
X_train = titanic_train[ features ]
y_train = titanic_train['Survived']
dt_estimator = tree.DecisionTreeClassifier()
dt_estimator.fit(X_train, y_train)
print(dt_estimator.tree_)
model_selection.cross_val_score(dt_estimator, X_train, y_train, scoring="accuracy", cv=10).mean()

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
dir = 'E:/'
graph.write_pdf(os.path.join(dir, "tree.pdf"))

titanic_test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(titanic_test.info())

#fit must only be performed only on train data since it must reflect the learning from train data only
titanic_test['Sex_encoded'] = lencoder.transform(titanic_test['Sex'])
titanic_test['Age_imputed'] = imp.transform(titanic_test[['Age']])

X_test = titanic_test[features]
titanic_test['Survived'] = dt_estimator.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)
