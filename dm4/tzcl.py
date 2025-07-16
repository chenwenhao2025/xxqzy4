import pandas as pd

def prepare_features(df):
    """准备模型特征：执行独热编码并拼接特征"""
    # 确保数据是字符串类型
    for col in ['Variety', 'City Name', 'Package', 'Origin']:
        df[col] = df[col].astype(str)
    
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
    X = pd.concat(features, axis=1)
    
    # 修复特征名称中的空格问题
    X.columns = [col.replace(' ', '_') for col in X.columns]
    
    return X