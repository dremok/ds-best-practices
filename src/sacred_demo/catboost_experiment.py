import pandas as pd
from catboost import Pool, cv
from sacred import Experiment

from sacred_demo.observers import file_observer, mongo_observer

house_prices_catboost = Experiment('house_prices_xgboost')

house_prices_catboost.observers.append(file_observer)
house_prices_catboost.observers.append(mongo_observer)


@house_prices_catboost.config
def my_config():
    iterations = 1000
    depth = 2
    params = {"iterations": iterations,
              "depth": depth,
              "loss_function": "MAE",
              "verbose": False}

    numerical_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                      'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                      'GrLivArea', 'BsmtFullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
                      'WoodDeckSF', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    categorical_cols = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'Exterior1st', 'Foundation',
                        'SaleCondition']
    target_col = 'SalePrice'


@house_prices_catboost.automain
def main(params, numerical_cols, categorical_cols, target_col):
    train_df = pd.read_csv('train.csv')
    train_df = train_df[numerical_cols + categorical_cols + [target_col]]
    cv_dataset = Pool(data=train_df.drop(target_col, axis=1),
                      label=train_df[target_col],
                      cat_features=categorical_cols)

    scores = cv(cv_dataset,
                params,
                fold_count=5)
    for i in range(len(scores)):
        house_prices_catboost.log_scalar('mae', scores['test-MAE-mean'][i], i)
    return scores['test-MAE-mean'][999]
