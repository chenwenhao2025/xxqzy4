import pandas as pd

def load_data(file_path):
    """加载数据集并提取关键特征"""
    df = pd.read_csv(file_path)
    features = ['Date', 'City Name', 'Origin', 'Variety', 'Package', 'Low Price', 'High Price']
    return df[features]

def preprocess_data(data):
    """数据预处理：添加时间特征和价格特征"""
    # 提取年份
    data['Year'] = data['Date'].apply(lambda dt: pd.to_datetime(dt).year)

    # 提取月份
    data['Month'] = pd.to_datetime(data['Date'], errors='coerce').dt.month
    
    # 提取年中的天数
    data['DayOfYear'] = pd.to_datetime(data['Date'], errors='coerce').dt.dayofyear
    
    # 计算平均价格
    data['Price'] = (data['Low Price'] + data['High Price']) / 2
    
    # 选择新特征集
    new_features = ['Year', 'Month', 'DayOfYear', 'City Name', 'Origin', 'Variety', 'Package', 'Price']
    processed = data[new_features].reset_index(drop=True)
    
    # 删除缺失值
    processed.dropna(inplace=True)
    return processed