from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

def train_xgboost_model(X_train, y_train):
    """训练XGBoost模型并进行参数调优"""
    model = XGBRegressor(random_state=42)
    
    # 设置参数网格
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 6],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"XGBoost最优参数: {grid_search.best_params_}")
    print(f"XGBoost最优分数: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_