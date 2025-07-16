from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    """训练随机森林模型并进行参数调优"""
    # 使用默认参数初始化模型
    rf = RandomForestRegressor(random_state=42)
    
    # 设置参数网格用于搜索
    param_grid = {
        'n_estimators': [100, 200],  # 树的数量
        'max_depth': [None, 10, 20],  # 树的最大深度
        'min_samples_split': [2, 5],  # 分割内部节点所需的最小样本数
        'min_samples_leaf': [1, 2]    # 叶节点所需的最小样本数
    }
    
    # 创建GridSearchCV对象进行参数优化
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,  # 3折交叉验证
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # 执行网格搜索
    grid_search.fit(X_train, y_train)
    
    print(f"随机森林最优参数: {grid_search.best_params_}")
    print(f"随机森林最优分数: {-grid_search.best_score_:.4f}")
    
    # 返回最优模型
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_