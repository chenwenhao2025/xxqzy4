# 南瓜价格预测

```text
项目目录/
|-- README.md
|-- dm3/
|   |-- xxqdm2.2.ipynb
|   |-- xxqdm2.2.py
|-- dm4/
|   |-- sjycl.py
|   |-- ksh.py
|   |-- tzcl.py
|   |-- xx_mx.py
|   |-- sjsl_mx.py
|   |-- LGBM_mx.py
|   |-- XGBoost_mx.py
|   |-- main.py
|-- kshtp/
|   |-- city_price_distribution.png
|   |-- dayofyear_price_trend.png
|   |-- monthly_price_trend.png
|   |-- origin_price_distribution.png
|   |-- package_price_distribution.png
|   |-- price_distribution.png
|   |-- variety_price_distribution.png
|-- sc/
|   |-- tree_visualizations/
|   |   |-- rf_tree_1_of_5.png
|   |   |-- rf_tree_2_of_5.png
|   |   |-- rf_tree_2_polt.png
|   |   |-- rf_tree_3_of_5.png
|   |   |-- rf_tree_4_of_5.png
|   |   |-- rf_tree_5_of_5.png
|   |   |-- rf_tree_29_plot.png
|   |   |-- rf_tree_39_plot.png
|   |   |-- rf_tree_96_plot.png
|   |   |-- rf_tree_119_plot.png
|   |-- LGBM_metrics.txt
|   |-- LinearRegression_metrics.txt
|   |-- RandomForest_metrics.txt
|   |-- XGBoost_metrics.txt
|-- sj/
|   |-- US-pumpkins.csv
```


（1）列举你对南瓜数据掌握的细节：

比如，日期信息，在数据中没有很多参考价值，提供证据
<img width="844" height="544" alt="monthly_price_trend" src="https://github.com/user-attachments/assets/db408fd2-8088-4165-aa9e-c17f30717064" />

<img width="844" height="545" alt="dayofyear_price_trend" src="https://github.com/user-attachments/assets/343f2f32-090f-4db7-a244-f543bb82462b" />


比如，数据中每种种类，规格，城市，产地 样本分布怎么样？哪里样本充分等

<img width="843" height="604" alt="city_price_distribution" src="https://github.com/user-attachments/assets/6d9938b5-5c6f-4e08-8792-02a530667ba2" />


<img width="856" height="610" alt="origin_price_distribution" src="https://github.com/user-attachments/assets/e7a96a9c-1bd7-4fa2-aa0a-77d5bca8dd10" />


<img width="843" height="658" alt="variety_price_distribution" src="https://github.com/user-attachments/assets/e63a4b3c-026f-4aa8-86da-c237bfd3a30e" />


<img width="854" height="640" alt="package_price_distribution" src="https://github.com/user-attachments/assets/c4fdb2c2-cfdd-4c3c-97af-a3a84664178a" />


<img width="848" height="544" alt="price_distribution" src="https://github.com/user-attachments/assets/e1221a21-a355-45f5-a92e-1486ba8f7547" />



（2）列举你对特征处理掌握的细节：

比如，特征热力图，可分析特征线性相关性；而即使存在多重共线性，也不影响模型建模，可能会存在性能上的影响；

比如，线性模型对离散值的顺序编码非常敏感，树模型会好一些，但是也会有影响；

比如，删掉年月日信息，模型性能会减弱一些



（3）列举你对模型掌握的细节：

比如 树模型，树的深度，以及树的最大划分数，直接决定模型性能以及 过拟合的可能

比如 模型从统计中，学到的智能/答案
