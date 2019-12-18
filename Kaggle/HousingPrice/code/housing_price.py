# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Path of the file to read
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
test_data = pd.read_csv("../input/test.csv")
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
test_X =  test_data[features]

# Split into validation and training data
#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
#iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
#iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
#val_predictions = iowa_model.predict(val_X)
#val_mae = mean_absolute_error(val_predictions, val_y)
#print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
#iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
#iowa_model.fit(train_X, train_y)
#val_predictions = iowa_model.predict(val_X)
#val_mae = mean_absolute_error(val_predictions, val_y)
#print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


# Set up code checking
#from learntools.core import binder
#binder.bind(globals())
#from learntools.machine_learning.ex6 import *
#print("\nSetup complete")

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor()

# fit your model
rf_model.fit(X, y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_predictions = rf_model.predict(test_X)
#rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
#print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
#print(rf_val_predictions)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': rf_val_predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

