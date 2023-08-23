# -*-codeing = utf-8-*-
# @Time:2023/8/1614:34
# @Author:张旭
# @File:STL11.py
# @Software:PyCharm
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib as mpl







plt.rc("figure", figsize=(10, 6))
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
data = pd.read_csv('D:\desktop\Rstl论文\修改意见\夏威夷月度.csv', parse_dates=['date'], index_col='date')




print(data)
'''data.plot()
plt.ylabel('客流量')
plt.xlabel('日期')
plt.show()'''
res = STL(data,period=12).fit()
res.plot()
data['trend'] = res.trend

data['seasonal'] = res.seasonal
data['resid'] = res.resid
print('residual mean:',data.resid.mean())
data.resid.hist()
plt.show()