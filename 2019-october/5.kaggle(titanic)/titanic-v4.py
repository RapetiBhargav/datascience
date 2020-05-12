import pandas as pd
import os
import numpy as np
from sklearn import neighbors, model_selection, preprocessing

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
X = titanic_train[ features ]
y = titanic_train['Survived']

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)

#grid search based model building
knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors': list(range(2,10)) }
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, scoring='accuracy', cv=10)
knn_grid_estimator.fit(X_train, y_train)
print(knn_grid_estimator.best_params_)
print(knn_grid_estimator.best_score_)
print(knn_grid_estimator.score(X_train, y_train))

print(knn_grid_estimator.score(X_eval, y_eval))

titanic_test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(titanic_test.info())

titanic_test['Sex_encoded'] = lencoder.transform(titanic_test['Sex'])
titanic_test['Age_imputed'] = imp.transform(titanic_test[['Age']])

X_test = titanic_test[features]
titanic_test['Survived'] = knn_grid_estimator.best_estimator_.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)
