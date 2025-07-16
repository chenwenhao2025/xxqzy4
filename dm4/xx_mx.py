from sklearn.linear_model import LinearRegression

def train_linear_model(X_train, y_train):
    """训练线性回归模型"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model