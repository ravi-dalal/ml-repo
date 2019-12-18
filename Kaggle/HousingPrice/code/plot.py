import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence

training_data = pd.read_csv('../input/train.csv')
y = training_data.SalePrice

cols_to_use = ['LotArea', 'YearBuilt', 'OverallQual']
X = training_data[cols_to_use]

my_imputer = Imputer()
train_X = my_imputer.fit_transform(X)

my_model = GradientBoostingRegressor()
my_model.fit(train_X, y)
my_plots = plot_partial_dependence(my_model,       
                                   features=[0, 2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=cols_to_use, # labels on graphs
                                   grid_resolution=10)