import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
plt.rcParams['axes.unicode_minus'] = False    # 修复负号显示

os.makedirs('../kshtp', exist_ok=True)  # 创建保存可视化图片的文件夹

def save_figure(fig, filename):
    """保存图表到kshtp文件夹"""
    fig.savefig(f'../kshtp/{filename}', bbox_inches='tight')
    plt.close(fig)  # 关闭图形释放内存

def plot_data(data):
    """绘制数据分布图并保存到kshtp文件夹"""
    # 绘制价格分布图
    fig1 = plt.figure(figsize=(10, 6))
    sns.histplot(data['Price'], kde=True, bins=30)
    plt.title('价格分布图')
    plt.xlabel('价格')
    plt.ylabel('频数')
    save_figure(fig1, 'price_distribution.png')

    # 绘制月份价格趋势图
    fig2 = plt.figure(figsize=(10, 6))
    sns.lineplot(x='Month', y='Price', data=data)
    plt.title('平均价格随月份变化趋势图')
    plt.xlabel('月份')
    plt.ylabel('平均价格')
    save_figure(fig2, 'monthly_price_trend.png')

    # 绘制城市价格分布图
    fig3 = plt.figure(figsize=(10, 6))
    sns.boxplot(x='City Name', y='Price', data=data)
    plt.title('不同城市的价格分布图')
    plt.xlabel('城市')
    plt.ylabel('价格')
    plt.xticks(rotation=45)
    save_figure(fig3, 'city_price_distribution.png')

    # 绘制品种价格分布图
    fig4 = plt.figure(figsize=(10, 6))
    sns.boxplot(x='Variety', y='Price', data=data)
    plt.title('不同品种的价格分布图')
    plt.xlabel('品种')
    plt.ylabel('价格')
    plt.xticks(rotation=45)
    save_figure(fig4, 'variety_price_distribution.png')

    # 绘制包装价格分布图
    fig5 = plt.figure(figsize=(10, 6))
    sns.boxplot(x='Package', y='Price', data=data)
    plt.title('不同包装的价格分布图')
    plt.xlabel('包装')
    plt.ylabel('价格')
    plt.xticks(rotation=45)
    save_figure(fig5, 'package_price_distribution.png')

    # 绘制原产地价格分布图
    fig6 = plt.figure(figsize=(10, 6))
    sns.boxplot(x='Origin', y='Price', data=data)
    plt.title('不同原产地的价格分布图')
    plt.xlabel('原产地')
    plt.ylabel('价格')
    plt.xticks(rotation=45)
    save_figure(fig6, 'origin_price_distribution.png')

    # 绘制年中的天数价格趋势图
    fig7 = plt.figure(figsize=(10, 6))
    sns.lineplot(x='DayOfYear', y='Price', data=data)
    plt.title('年中的天数价格趋势图')
    plt.xlabel('年中的天数')
    plt.ylabel('平均价格')
    save_figure(fig7, 'dayofyear_price_trend.png')