import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

training_data = pd.read_csv('../input/train.csv')
#test_data = pd.read_csv('../input/test.csv')
training_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = training_data.SalePrice

X = training_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
#test_X = my_imputer.fit_transform(test_data.select_dtypes(exclude=['object']))
#print(train_X.shape)

#my_model = XGBRegressor()
#my_model.fit(train_X, train_y, verbose=False)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5,
             eval_set=[(test_X, test_y)],
			 verbose=False)
predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))