import pandas as pd
from sacred import Experiment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sacred_demo.observers import file_observer, mongo_observer

house_prices_sklearn = Experiment('house_prices_sklearn')

house_prices_sklearn.observers.append(file_observer)
house_prices_sklearn.observers.append(mongo_observer)


@house_prices_sklearn.config
def my_config():
    n_estimators = 10
    numerical_imputation = 'median'

    numerical_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                      'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                      'GrLivArea', 'BsmtFullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
                      'WoodDeckSF', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    categorical_cols = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'Exterior1st', 'Foundation',
                        'SaleCondition']
    target_col = 'SalePrice'

    numerical_transformer = SimpleImputer(strategy=numerical_imputation)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)

    pipeline = Pipeline([
        ('preprocesser', preprocessor),
        ('model', model)
    ])


@house_prices_sklearn.automain
def main(pipeline, target_col):
    train_df = pd.read_csv('train.csv')
    mae = -cross_val_score(pipeline, train_df.drop(target_col, axis=1), train_df[target_col], cv=5,
                           scoring='neg_mean_absolute_error').mean()
    house_prices_sklearn.log_scalar('mae', mae)
    return mae
