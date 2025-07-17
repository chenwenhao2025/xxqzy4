# 南瓜价格预测

项目目录/
│  README.md
├── dm3/
│   ├── xxqdm2.2.ipynb
│   └── xxqdm2.2.py
├── dm4/
│   ├── sjycl.py
│   ├── ksh.py
│   ├── tzcl.py
│   ├── xx_mx.py
│   ├── sjsl_mx.py
│   ├── LGBM_mx.py
│   ├── XGBoost_mx.py
│   └── main.py
├── kshtp/
│   ├── city_price_distribution.png
│   ├── dayofyear_price_trend.png
│   ├── monthly_price_trend.png
│   ├── origin_price_distribution.png
│   ├── package_price_distribution.png
│   ├── price_distribution.png
│   └── variety_price_distribution.png
├── sc/
│   ├── tree_visualizations/
│   │   ├── rf_tree_1_of_5.png
│   │   ├── rf_tree_2_of_5.png
│   │   ├── rf_tree_2_polt.png
│   │   ├── rf_tree_3_of_5.png
│   │   ├── rf_tree_4_of_5.png
│   │   ├── rf_tree_5_of_5.png
│   │   ├── rf_tree_29_plot.png
│   │   ├── rf_tree_39_plot.png
│   │   ├── rf_tree_96_plot.png
│   │   └── rf_tree_119_plot.png
│   ├── LGBM_metrics.txt
│   ├── LinearRegression_metrics.txt
│   ├── RandomForest_metrics.txt
│   └── XGBoost_metrics.txt
└── sj/
   └── US-pumpkins.csv



（1）列举你对南瓜数据 掌握的 细节：

比如，日期信息，在数据中没有很多参考价值，提供证据

比如，数据中每种种类，规格，城市，产地 样本分布怎么样？哪里样本充分等



（2）列举你对特征处理 掌握的 细节：

比如，特征热力图，可分析特征线性相关性；而即使存在多重共线性，也不影响模型建模，可能会存在性能上的影响；

比如，线性模型对离散值的顺序编码非常敏感，树模型会好一些，但是也会有影响；

比如，删掉年月日信息，模型性能会减弱一些



（3）列举你对 模型 掌握的 细节：

比如 树模型，树的深度，以及树的最大划分数，直接决定模型性能以及 过拟合的可能

比如 模型从统计中，学到的智能/答案
