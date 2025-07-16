import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dm4.sjycl import load_data, preprocess_data
from dm4.ksh import plot_data
from dm4.tzcl import prepare_features
from dm4.xx_mx import train_linear_model
from dm4.sjsl_mx import train_random_forest
from dm4.LGBM_mx import train_lgbm_model
from dm4.XGBoost_mx import train_xgboost_model
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_graphviz
import graphviz
import random


warnings.filterwarnings('ignore')

# 确保输出目录存在
os.makedirs('../sc', exist_ok=True)
os.makedirs('../sc/tree_visualizations', exist_ok=True)  # 创建树可视化目录


def save_metrics(model_name, metrics):
    """保存模型指标到文件"""
    with open(f'../sc/{model_name}_metrics.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def evaluate_model(model, X_test, y_test):
    """评估模型性能并返回指标"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2


def plot_model_performance(model, X_train, y_train, X_test, y_test):
    """计算训练集和测试集的R²分数"""
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    return train_r2, test_r2


def visualize_random_forest_trees(model, feature_names, max_trees=5, max_depth=3):
    """
    可视化随机森林中的决策树
    Args:
        model: 训练好的随机森林模型
        feature_names: 特征名称列表
        max_trees: 最多可视化的树的数量
        max_depth: 可视化树的最大深度
    """
    print("\n生成随机森林树状图...")
    # 获取随机森林中的树
    estimators = model.estimators_
    n_trees = min(max_trees, len(estimators))

    # 随机选择要可视化的树
    selected_indices = random.sample(range(len(estimators)), n_trees)

    plt.figure(figsize=(20, 12))

    # 为每棵树创建可视化
    for i, idx in enumerate(selected_indices):
        tree = estimators[idx]

        # 创建树的图形表示
        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=feature_names,
            class_names=['Price'],  # 回归任务没有类别，但需要这个参数
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=max_depth,  # 限制深度以避免过于复杂
            proportion=True,
            impurity=False
        )

        # 使用graphviz创建图形
        graph = graphviz.Source(dot_data)

        # 保存为PNG文件
        filename = f"../sc/tree_visualizations/rf_tree_{i + 1}_of_{n_trees}.png"
        graph.render(filename=filename.replace('.png', ''), format='png', cleanup=True)
        print(f"已保存树可视化: {filename}")

        # 生成并保存树状图图片
        plt.figure(figsize=(15, 10))
        plot_tree(tree,
                  feature_names=feature_names,
                  filled=True,
                  max_depth=3,  # 限制显示深度
                  proportion=True,
                  rounded=True,
                  fontsize=8)
        plt.title(f'随机森林 - 树 #{idx + 1}/{len(estimators)}')
        plot_filename = f"../sc/tree_visualizations/rf_tree_{idx + 1}_plot.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存树状图: {plot_filename}")



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

    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {len(X)}")

    # 检查数据量是否足够
    if len(X) < 1000:
        print(f"样本数量较少 ({len(X)})，可能影响模型性能")
        # 考虑减少测试集比例
        test_size = 0.1
    else:
        test_size = 0.2

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # ==========================================
    # 线性回归模型
    # ==========================================
    print("\n" + "=" * 40)
    print("训练线性回归模型")
    print("=" * 40)
    lr_model = train_linear_model(X_train, y_train)

    # 模型评估
    lr_mse, lr_rmse, lr_r2 = evaluate_model(lr_model, X_test, y_test)
    lr_train_r2, lr_test_r2 = plot_model_performance(
        lr_model, X_train, y_train, X_test, y_test
    )

    # 保存指标
    lr_metrics = {
        'MSE': lr_mse,
        'RMSE': lr_rmse,
        'R2': lr_r2,
        'Train_R2': lr_train_r2
    }
    save_metrics('LinearRegression', lr_metrics)

    # ==========================================
    # 随机森林模型
    # ==========================================
    print("\n" + "=" * 40)
    print("训练随机森林模型")
    print("=" * 40)
    rf_model, rf_best_params, rf_best_score = train_random_forest(X_train, y_train)

    # 模型评估
    rf_mse, rf_rmse, rf_r2 = evaluate_model(rf_model, X_test, y_test)
    rf_train_r2, rf_test_r2 = plot_model_performance(
        rf_model, X_train, y_train, X_test, y_test
    )

    # 保存指标
    rf_metrics = {
        'best_params': rf_best_params,
        'best_score': rf_best_score,
        'MSE': rf_mse,
        'RMSE': rf_rmse,
        'R2': rf_r2,
        'Train_R2': rf_train_r2
    }
    save_metrics('RandomForest', rf_metrics)

    # 获取特征名称
    feature_names = X.columns.tolist()

    # 可视化随机森林的决策树
    visualize_random_forest_trees(rf_model, feature_names)

    # ==========================================
    # LGBM模型
    # ==========================================
    print("\n" + "=" * 40)
    print("训练LGBM模型")
    print("=" * 40)
    lgb_model, lgb_best_params, lgb_best_score = train_lgbm_model(X_train, y_train)

    # 模型评估
    lgb_mse, lgb_rmse, lgb_r2 = evaluate_model(lgb_model, X_test, y_test)
    lgb_train_r2, lgb_test_r2 = plot_model_performance(
        lgb_model, X_train, y_train, X_test, y_test
    )

    # 保存指标
    lgb_metrics = {
        'best_params': lgb_best_params,
        'best_score': lgb_best_score,
        'MSE': lgb_mse,
        'RMSE': lgb_rmse,
        'R2': lgb_r2,
        'Train_R2': lgb_train_r2
    }
    save_metrics('LGBM', lgb_metrics)

    # ==========================================
    # XGBoost模型
    # ==========================================
    print("\n" + "=" * 40)
    print("训练XGBoost模型")
    print("=" * 40)
    xgb_model, xgb_best_params, xgb_best_score = train_xgboost_model(X_train, y_train)

    # 模型评估
    xgb_mse, xgb_rmse, xgb_r2 = evaluate_model(xgb_model, X_test, y_test)
    xgb_train_r2, xgb_test_r2 = plot_model_performance(
        xgb_model, X_train, y_train, X_test, y_test
    )

    # 保存指标
    xgb_metrics = {
        'best_params': xgb_best_params,
        'best_score': xgb_best_score,
        'MSE': xgb_mse,
        'RMSE': xgb_rmse,
        'R2': xgb_r2,
        'Train_R2': xgb_train_r2
    }
    save_metrics('XGBoost', xgb_metrics)


if __name__ == "__main__":
    main()