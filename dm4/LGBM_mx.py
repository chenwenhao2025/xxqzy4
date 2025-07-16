import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

def train_lgbm_model(X_train, y_train):
    """训练LGBM模型并进行参数调优"""
    model = lgb.LGBMRegressor(random_state=42, force_row_wise=True, verbose=-1)
    
    # 设置参数网格
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [15, 41],
        'max_depth': [5, 10],
        'min_child_samples': [20, 50]
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
    
    print(f"LGBM最优参数: {grid_search.best_params_}")
    print(f"LGBM最优分数: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_