import pandas as pd
from PIL import Image
import numpy as np
import  five1
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import stats
def map2(file_path,option):
    # 读取Excel文件 # 请替换为你的文件路径
    df = pd.read_excel(file_path)
    font_path = '仿宋_GB2312.ttf'
    font = FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
    # 获取最后两列数据
    x = df.iloc[:, -1].values  # 最后一列作为X轴
    y = df.iloc[:, -2].values  # 倒数第二列作为Y轴

    # 进行线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # 计算 R² 值
    r_squared = r_value**2

    # 生成拟合的Y值
    y_pred = slope * x + intercept

    # 绘制散点图
    plt.scatter(x, y, color='blue', label='数据点')

    # 绘制回归线
    plt.plot(x, y_pred, color='black', label='回归线')

    # 在图中显示回归方程和 R²
    equation = f'y = {slope:.4f}x + {intercept:.4f}\n$R^2$ = {r_squared:.4f}'
    plt.text(min(x), max(y), equation, fontsize=12, color='black', verticalalignment='top')

    # 添加标题和标签
    plt.xlabel('均匀度（预测）',fontproperties=font)
    plt.ylabel('功率密度',fontproperties=font)
    plt.title('线性回归分析',fontproperties=font)
    plt.grid(False)
    # 保存图像
    save_path = f'{option}_materials_comparison1.png'  # 你可以更改文件名或路径
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存为 {save_path}")
    # # 显示图形
    # plt.show()
    plt.show()
files = [
     "data.xlsx"
]
five1.five1(files)
map2("data.xlsx","test02")
