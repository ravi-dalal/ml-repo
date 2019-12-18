# Read the data
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('../input/train.csv')
#test_data = pd.read_csv('../input/test.csv')

# Drop houses where the target is missing
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
#cols_with_missing = [col for col in train_data.columns 
#                                if train_data[col].isnull().any()]
train_X = train_data.drop(['Id', 'SalePrice'], axis=1)
y = train_data.SalePrice
#test_X = test_data.drop(['Id'], axis=1)
low_cardinality_cols = [cname for cname in train_X.columns if 
                                train_X[cname].nunique() < 10 and
                                train_X[cname].dtype == "object"]
numeric_cols = [cname for cname in train_X.columns if 
                                train_X[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols

train_predictors = train_X[my_cols]
#test_predictors = test_X[my_cols]
#print(train_predictors.shape)
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
#one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
#print(*one_hot_encoded_training_predictors.columns, sep=',')
#my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': rf_val_predictions})
#one_hot_encoded_training_predictors.to_csv('TRAINING_1.csv', index=False)
#final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
#                                                                    join='left', 
#                                                                    axis=1)
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
#my_pipeline.fit(final_train, y)
#print(my_pipeline.predict(final_test))
scores = cross_val_score(my_pipeline, one_hot_encoded_training_predictors, y, scoring='neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * scores.mean()))

