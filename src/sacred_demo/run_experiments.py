from sacred_demo.catboost_experiment import house_prices_catboost
from sacred_demo.sklearn_experiment import house_prices_sklearn

random_forest_default = house_prices_sklearn.run()
random_forest_more_estimators = house_prices_sklearn.run(config_updates={'n_estimators': 50})
random_forest_mean_imputation = house_prices_sklearn.run(
    config_updates={'numerical_imputation': 'mean', 'n_estimators': 50})

catboost_default = house_prices_catboost.run()
catboost_increased_depth = house_prices_catboost.run(config_updates={'depth': 3})
