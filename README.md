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

 业务解释：为什么日期无效？
​​供需非时间主导​​价格波动主要受突发事件驱动
​​（如：某产地暴雨（供给中断），节日临时采购（需求激增））而非日期本身。
​​库存策略干扰​​经销商在200天（年中）的​​集体抛售库存​​行为导致价格反逻辑暴跌，打破时间规律。
​​品种成熟周期差异​​不同南瓜品种在第200天时：
早熟种已过剩（降价）
晚熟种刚上市（涨价）→ 日期无法反映混合品种的真实供需


比如，数据中每种种类，规格，城市，产地样本分布怎么样？哪里样本充分等

<img width="843" height="604" alt="city_price_distribution" src="https://github.com/user-attachments/assets/6d9938b5-5c6f-4e08-8792-02a530667ba2" />

“不同城市的价格分布图” 中，各城市箱线图差异明显。如 BALTIMORE 价格离散程度大，有极端值；ATLANTA 价格集中在低位。说明不同城市样本，价格分布特征不同，可能是各城市样本数量、当地市场特点等导致，像 BALTIMORE 可能有多样价格样本，ATLANTA 样本价格相对单一。

<img width="856" height="610" alt="origin_price_distribution" src="https://github.com/user-attachments/assets/e7a96a9c-1bd7-4fa2-aa0a-77d5bca8dd10" />

“不同原产地的价格分布图” 里，各原产地箱线图形态各异。如 NORTH CAROLINA 价格整体偏高且箱线图 “箱体” 长，离散大；TEXAS 价格多集中在低位。反映不同原产地样本，价格分布有别，推测是原产地供货、品质等因素，使样本在价格上呈现不同分布，比如 NORTH CAROLINA 样本价格多样，TEXAS 样本价格相对集中。

<img width="843" height="658" alt="variety_price_distribution" src="https://github.com/user-attachments/assets/e63a4b3c-026f-4aa8-86da-c237bfd3a30e" />

从 “不同品种的价格分布图” 箱线图看，不同品种如 HOWDEN TYPE、CINDERELLA 等，箱线图形态差异大。像 MINIATURE 品种，价格集中在极低区间，且离散程度小；而 PIE TYPE 等品种，价格区间跨度大，说明不同品种样本在价格表现上分布不同，推测样本数量或特征有别，比如 MINIATURE 可能样本价格相对统一，PIE TYPE 样本价格多样。

<img width="854" height="640" alt="package_price_distribution" src="https://github.com/user-attachments/assets/c4fdb2c2-cfdd-4c3c-97af-a3a84664178a" />

“不同包装的价格分布图” 里，24 inch bins、36 inch bins 等包装，箱线图高矮、离散程度不同。24 inch bins、36 inch bins 价格整体偏高且离散大，说明这些包装对应的样本中价格波动大；像 1/9 bushel cartons 等小包装，价格集中在低位，样本价格相对整齐，反映不同包装规格下，样本在价格维度的分布特征，间接体现样本数量或构成差异（若样本多且杂，离散易大 ）。

<img width="848" height="544" alt="price_distribution" src="https://github.com/user-attachments/assets/e1221a21-a355-45f5-a92e-1486ba8f7547" />

从 “价格分布图”（直方图 + 核密度曲线 ）来看，样本价格分布呈现以下特点：
1. 多峰分布
有明显多个峰值，在价格 0 - 50 区间、150 - 200 区间等存在峰值，说明价格并非集中在单一区间，而是不同价格段都有较多样本分布，数据分布相对复杂，可能对应不同产品类型、规格或市场场景下的价格表现 。
2. 区间差异大
价格覆盖 0 - 500 区间，低价格段（0 - 100 左右 ）和中价格段（100 - 250 左右 ）样本频数高，是分布主要区域；高价格段（300 以上 ）样本频数极低，说明大部分样本价格集中在中低区间，高价格样本占比少，整体价格分布偏向左端（低价格侧 ）。
3. 离散程度
从核密度曲线看，数据离散程度较大，不同价格段都有分布，不过主要集中在 0 - 250 区间内，该区间外（尤其是 300 以上 ）样本稀疏，呈现长尾分布特征，即少数高价格样本拉长篇尾 。


（2）列举你对特征处理掌握的细节：

比如，特征热力图，可分析特征线性相关性；而即使存在多重共线性，也不影响模型建模，可能会存在性能上的影响；

比如，线性模型对离散值的顺序编码非常敏感，树模型会好一些，但是也会有影响；

比如，删掉年月日信息，模型性能会减弱一些



（3）列举你对模型掌握的细节：

比如，树模型，树的深度，以及树的最大划分数，直接决定模型性能以及 过拟合的可能

比如，模型从统计中，学到的智能/答案











