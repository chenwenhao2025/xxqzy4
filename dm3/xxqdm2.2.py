import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
plt.rcParams['axes.unicode_minus'] = False    # 修复负号显示

def load_data(file_path):
    """加载数据集并提取关键特征"""
    df = pd.read_csv(file_path)
    features = ['Date', 'City Name', 'Origin', 'Variety', 'Package', 'Low Price', 'High Price']
    return df[features]

def preprocess_data(data):
    """数据预处理：添加时间特征和价格特征"""
    # 提取月份
    data['Month'] = data['Date'].apply(lambda dt: pd.to_datetime(dt).month)
    
    # 提取年中的天数
    data['DayOfYear'] = data['Date'].apply(lambda dt: pd.to_datetime(dt).timetuple().tm_yday)
    
    # 计算平均价格
    data['Price'] = (data['Low Price'] + data['High Price']) / 2
    
    # 选择新特征集
    new_features = ['Month', 'DayOfYear', 'City Name', 'Origin', 'Variety', 'Package', 'Price']
    processed = data[new_features].reset_index(drop=True)
    
    # 删除缺失值
    processed.dropna(inplace=True)
    return processed

def plot_data(data):
    """绘制数据分布图"""
    # 绘制价格分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Price'], kde=True, bins=30)
    plt.title('价格分布图')
    plt.xlabel('价格')
    plt.ylabel('频数')
    plt.show()

    # 绘制月份价格趋势图
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Month', y='Price', data=data)
    plt.title('平均价格随月份变化趋势图')
    plt.xlabel('月份')
    plt.ylabel('平均价格')
    plt.show()

    # 绘制城市价格分布图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='City Name', y='Price', data=data)
    plt.title('不同城市的价格分布图')
    plt.xlabel('城市')
    plt.ylabel('价格')
    plt.xticks(rotation=45)
    plt.show()

    # 绘制品种价格分布图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Variety', y='Price', data=data)
    plt.title('不同品种的价格分布图')
    plt.xlabel('品种')
    plt.ylabel('价格')
    plt.xticks(rotation=45)
    plt.show()

    # 绘制包装价格分布图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Package', y='Price', data=data)
    plt.title('不同包装的价格分布图')
    plt.xlabel('包装')
    plt.ylabel('价格')
    plt.xticks(rotation=45)
    plt.show()

    # 绘制原产地价格分布图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Origin', y='Price', data=data)
    plt.title('不同原产地的价格分布图')
    plt.xlabel('原产地')
    plt.ylabel('价格')
    plt.xticks(rotation=45)
    plt.show()

    # 绘制年中的天数价格趋势图
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='DayOfYear', y='Price', data=data)
    plt.title('年中的天数价格趋势图')
    plt.xlabel('年中的天数')
    plt.ylabel('平均价格')
    plt.show()

    return data

def prepare_features(df):
    """准备模型特征：执行独热编码并拼接特征"""
    # 对分类变量执行独热编码
    variety_dummies = pd.get_dummies(df['Variety'], prefix='variety')
    city_dummies = pd.get_dummies(df['City Name'], prefix='city')
    package_dummies = pd.get_dummies(df['Package'], prefix='package')
    origin_dummies = pd.get_dummies(df['Origin'], prefix='origin')
    
    # 添加月份特征
    month_dummies = pd.get_dummies(df['Month'], prefix='month').astype(int)
    
    # 拼接数值特征和独热编码特征
    features = [
        df['DayOfYear'], 
        month_dummies,
        variety_dummies, 
        city_dummies, 
        package_dummies, 
        origin_dummies
    ]
    
    # 确保所有特征都是数值型
    return pd.concat(features, axis=1)

def train_linear_model(X_train, y_train):
    """训练线性回归模型"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

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
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """评估模型性能并返回指标"""
    y_pred = model.predict(X_test)
    
    # 计算MSE和RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # 计算R²分数
    r2 = r2_score(y_test, y_pred)
    
    return mse, rmse, r2

def plot_model_performance(model, X_train, y_train, X_test, y_test):
    """
    模型在训练集和测试集上的表现
    """
    # 计算训练集的R²
    train_r2 = model.score(X_train, y_train)
   
    # 计算测试集的R²
    test_r2 = model.score(X_test, y_test)
    
    # 返回评估指标
    return {
        '训练集_R²': train_r2,
        '测试集_R²': test_r2
    }

def main():
    # 数据加载
    data = load_data("../sj/US-pumpkins.csv")
    
    # 数据预处理
    processed = preprocess_data(data)
    print(f"预处理后的数据量: {len(processed)}")
    
    # 数据可视化
    # plot_data(processed)
    
    # 特征工程
    X = prepare_features(processed)
    y = processed['Price']
    feature_names = list(X.columns)
    
    print(f"特征数量: {len(feature_names)}")
    print(f"样本数量: {len(X)}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ==========================================
    # 线性回归模型
    # ==========================================
    print("\n" + "="*40)
    lr_model = train_linear_model(X_train, y_train)
    
    # 模型评估
    lr_mse, lr_rmse, lr_r2 = evaluate_model(lr_model, X_test, y_test)
    
    # 查看训练集和测试集具体效果
    lr_perf = plot_model_performance(lr_model, X_train, y_train, X_test, y_test)
    
    # 打印结果
    print("\n线性回归评估结果:")
    print(f"MSE: {lr_mse:.2f}, RMSE: {lr_rmse:.2f}")
    print(f"测试集R²: {lr_r2:.4f}")
    print(f"训练集R²: {lr_perf['训练集_R²']:.4f}")
    
    # ==========================================
    # 随机森林模型
    # ==========================================
    print("\n" + "="*40)
    rf_model = train_random_forest(X_train, y_train)
    
    # 模型评估
    rf_mse, rf_rmse, rf_r2 = evaluate_model(rf_model, X_test, y_test)
    
    # 查看训练集和测试集具体效果
    rf_perf = plot_model_performance(rf_model, X_train, y_train, X_test, y_test)
    
    # 打印结果
    print("\n随机森林评估结果:")
    print(f"MSE: {rf_mse:.2f}, RMSE: {rf_rmse:.2f}")
    print(f"测试集R²: {rf_r2:.4f}")
    print(f"训练集R²: {rf_perf['训练集_R²']:.4f}")
    
    

if __name__ == "__main__":
    main()