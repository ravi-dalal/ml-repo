import pandas as pd
from sklearn.preprocessing import Imputer
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import re as re
import numpy as np

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""


training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
training_data.dropna(subset = ['Embarked'], inplace = True)
test_data.dropna(subset = ['Embarked'], inplace = True)
#print(training_data.info())
#print(test_data.info())
fare_data = training_data.dropna(subset = ["Fare"], inplace = False)[["Pclass","Embarked","Fare"]].groupby(["Pclass","Embarked"], as_index=False).mean()
full_data = [training_data, test_data]
#Fare
for dataset in full_data:
	for index, row in fare_data.iterrows():
		dataset.loc[(dataset['Fare'].isnull()) & (dataset['Pclass'] == row['Pclass']) & (dataset['Embarked'] == row['Embarked']), 'Fare'] = row['Fare']
#Family Size
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
#Cabin or not
for dataset in full_data:
    dataset['HasCabin'] = 1
    dataset.loc[dataset['Cabin'].isnull(), 'HasCabin'] = 0
#Alone or not
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
#Title
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#Age
age_data = training_data.dropna(subset = ["Age"], inplace = False)[["Title","Age"]].groupby(["Title"], as_index=False).mean()
for dataset in full_data:
	for index, row in age_data.iterrows():
		dataset.loc[(dataset['Age'].isnull()) & (dataset['Title'] == row['Title']), 'Age'] = row['Age']
#Cleanup
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

# Feature Selection
training_data.to_csv('modified_training_data.csv', index=False)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
train_X = training_data.drop(drop_elements, axis = 1)
test_X  = test_data.drop(drop_elements, axis = 1)
#training_data = training_data.values
#test_data  = test_data.values
y = train_X.Survived
X = train_X.drop(["Survived"], axis = 1)
'''
features = ["Pclass","Sex","Age","SibSp","Parch","Embarked"]
X = training_data[features]
test_X = test_data[features]
train_X = pd.get_dummies(X, columns=["Sex","Embarked"])
test_X = pd.get_dummies(test_X, columns=["Sex","Embarked"])
train_X = pd.get_dummies(X, columns=["Sex","Embarked","Title"])
print (train_X.head(10))
test_X = pd.get_dummies(test_data, columns=["Sex","Embarked","Title"])
'''

#my_pipeline = make_pipeline(XGBClassifier(max_depth=10, learning_rate=0.08, n_estimators=150, gamma=0.5))
my_pipeline = make_pipeline(SVC())
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * scores.mean()))
my_pipeline.fit(X, y)
predictions = my_pipeline.predict(test_X)
'''
my_pipeline = make_pipeline(Imputer(), KNeighborsClassifier())
scores = cross_val_score(my_pipeline, train_X, y, scoring='neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * scores.mean()))
my_pipeline.fit(train_X, y)
predictions = my_pipeline.predict(test_X)
my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, y, early_stopping_rounds=5,
             eval_set=[(train_X, y)],
			 verbose=False)
predictions = my_model.predict(test_X)
'''
my_submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
my_submission.to_csv('submission.csv', index=False)

