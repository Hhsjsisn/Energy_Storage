import time
from scipy import stats
import torch.optim as optim
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from PIL import Image
import torch.nn as nn
# # 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体的英文名是 SimSun
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def parse_excel_data(file_path):
    # 读取Excel文件，跳过第一行数据
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=None, skiprows=1)

    # 前向填充流道类型（第0列）
    df[0] = df[0].ffill()
    print(f"DataFrame的列数: {df.shape[1]}")
    # 提取数据
    # 输入特征：B(1)到Q(16)列，共17列（索引1到16）
    # 输出值：R(17)列
    x = df.iloc[:, 1:16].values  # 输入特征
    y = df.iloc[:, -1].values  # 取最后一列
    types = df.iloc[:, 0].values  # 流道类型
    return x, y, types
def parse_excel_data1(file_path):
    # 读取Excel文件，跳过第一行数据
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=None, skiprows=1)

    # 前向填充流道类型（第0列）
    df[0] = df[0].ffill()
    print(f"DataFrame的列数: {df.shape[1]}")
    # 提取数据
    # 输入特征：B(1)到Q(16)列，共17列（索引1到16）
    # 输出值：R(17)列
    x = df.iloc[:, 1:16].values  # 输入特征
    types = df.iloc[:, 0].values  # 流道类型
    return x, types
def map2(file_path,option):
    # 读取Excel文件 # 请替换为你的文件路径
    plt.figure()
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


# class StandardScaler(nn.Module):
#     def __init__(self):
#         super(StandardScaler, self).__init__()
#         self.mean = None
#         self.std = None
#
#     def fit(self, x):
#         self.mean = x.mean(0, keepdim=True)
#         self.std = x.std(0, keepdim=True) + 1e-7  # 添加小值避免除零
#         return self
#
#     def transform(self, x):
#         return (x - self.mean) / self.std
#
#     def fit_transform(self, x):
#         self.fit(x)
#         return self.transform(x)
class StandardScaler(nn.Module):
    def __init__(self):
        super(StandardScaler, self).__init__()
        self.mean = None
        self.std = None

    def fit(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)  # 确保 x 是 Tensor

        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True) + 1e-7  # 避免除零
        return self

    def transform(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
# 方法2：使用BatchNorm1d的神经网络
class RegressionNet(nn.Module):
    def __init__(self, input_size=15):
        super(RegressionNet, self).__init__()
        self.layers = nn.Sequential(
            #nn.BatchNorm1d(input_size),  # 添加BatchNorm层
            nn.Linear(input_size, 6),
            # nn.ReLU(),
            #  nn.Dropout(0.2),
            # nn.BatchNorm1d(64),  # 每层后添加BatchNorm
           # nn.Linear(64, 32),
            #  nn.ReLU(),
            #  nn.Dropout(0.2),
            # nn.BatchNorm1d(32),
            # nn.Linear(128, 64),
          #  nn.Linear(64,32),
            #nn.ReLU(),
            # nn.BatchNorm1d(16),
            # nn.Linear(96, 32),
            # # nn.Dropout(0.2),
            #  nn.Linear(32, 8),
            # nn.Linear(32, 8),
            #   nn.Linear(16, 8),
            # nn.Linear(32, 16),
            #  nn.Linear(16, 8),
            #  nn.ReLU(),
            # nn.Linear(8, 4),
            nn.Linear(6, 1),

            nn.Linear(1, 1),
        )

    def forward(self, x):
        return self.layers(x)


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_and_evaluate_model(X, y,scaler, device='cpu' if torch.cuda.is_available() else 'cpu'):
    all_data = []
    """训练和评估模型"""
    # 标准化特征
    X_tensor = torch.FloatTensor(X)
    # 如果没有传入scaler，创建新的scaler
    # if scaler is None:
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X_tensor)
    # else:
    #     print("使用传入的scaler进行转换")
    #     # 使用传入的scaler进行转换
    #     X_scaled = scaler.fit_transform(X_tensor)
    #创建数据加载器
    dataset = CustomDataset(X_tensor, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 初始化模型
    model = RegressionNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

    # 预测
    model.eval()

    for index in range(len(dataloader)):
        input = torch.FloatTensor(X[index])
        with torch.no_grad():
            outputs = model(input)
            targets = torch.tensor( y[index]).unsqueeze(0)


            output_np = outputs.detach().numpy()
            # 保持横坐标不变，不对输出进行反归一化
            x_data = output_np.flatten()  # 横坐标



            y_data =targets.detach().numpy()
            all_data.append((x_data, y_data))
            # 根据 index 分类流道类型

    # # 训练模型

    # with torch.no_grad():
    #     #x = torch.FloatTensor(X_scaled).to(device)
    #     # x = X_tensor.to(device)
    #     x = scaler.fit_transform(X_tensor)
    #     y_out = model(x)
    #     y_pred = y_out.cpu().numpy()
    #     output_np = outputs.detach().numpy()
    #     last_layer = model.layers[-1]  # 获取最后一层（Linear层）
    #     weights = last_layer.weight.cpu().detach().numpy()  # 权重矩阵
    #     bias = last_layer.bias.cpu().detach().numpy()  # 偏置向量
    #
    # # 将预测值转换为一维数组
    # y_pred = y_pred.reshape(-1)
    #
    # # 计算R²值
    # y_mean = np.mean(y)
    # ss_tot = np.sum((y - y_mean) ** 2)
    # ss_res = np.sum((y - y_pred) ** 2)
    # r2 = 1 - (ss_res / ss_tot)
    # print(f"R² Score: {r2:.4f}")
    # # 或者直接使用 sklearn 的 r2_score，确保输入是一维数组
    # r2_sklearn = r2_score(y, y_pred)
    # print(f"sklearn R² Score: {r2_sklearn:.4f}")
    # torch.save(model.state_dict(), 'regression_model.pth')
    # print("模型已保存为 regression_model.pth")
    #
    # return model, scaler, X, y, output_np, r2,weights,bias
    return all_data

def train_and_evaluate_model_l(X, y, scaler, device='cpu' if torch.cuda.is_available() else 'cpu'):
    # 转换数据为 Tensor
    X_tensor = torch.FloatTensor(X)
    dataset = CustomDataset(X_tensor, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 初始化模型
    model = RegressionNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100  # 训练周期
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'regression_model_l.pth')
    print("模型已保存为 regression_model.pth")
def predict_and_save(X_new, xlsx_path, model_path='regression_model_l.pth', device='cpu'):
    """加载已训练的模型，进行预测，并将结果保存到 Excel 文件的 'predict' 列"""
    model = RegressionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 确保 X_new 是 numpy 数组并转换为 torch.Tensor
    if isinstance(X_new, np.ndarray):
        X_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
    else:
        raise ValueError("输入数据 X_new 必须是 numpy.ndarray 类型")

    # 进行预测
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().flatten()  # 确保是 1D 数组

    # 读取 Excel 文件
    try:
        df = pd.read_excel(xlsx_path)  # 读取现有的 Excel 文件
    except FileNotFoundError:
        df = pd.DataFrame()  # 如果文件不存在，创建一个空 DataFrame

    # 确保 'predict' 列存在
    if 'Predict' not in df.columns:
        df['Predict'] = np.nan  # 先创建 'Predict' 列

    # 确保 Excel 行数足够
    num_rows_needed = len(predictions)
    if len(df) < num_rows_needed:
        additional_rows = pd.DataFrame({col: [np.nan] * (num_rows_needed - len(df)) for col in df.columns})
        df = pd.concat([df, additional_rows], ignore_index=True)  # 扩展行数

    # 填充预测值
    df.loc[:num_rows_needed - 1, 'Predict'] = predictions

    # 保存回 Excel 文件
    df.to_excel(xlsx_path, index=False)
    # print(f"预测值已保存到 {xlsx_path} 的 'predict' 列")


def plot_results_multiple(all_results,type):
    """绘制多个材料的散点和拟合直线"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    font_path = '仿宋_GB2312.ttf'
    font = FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
    # 点的形状样式

    point_styles = {
        '网格型流道': {'color': 'red', 'marker': 'o', 'label': '网格型流道'},
        '蛇形流道': {'color': 'blue', 'marker': 's', 'label': '蛇形流道'},
        '交指型流道': {'color': 'green', 'marker': 'x', 'label': '交指型流道'},
        '直平行流道': {'color': 'purple', 'marker': '*', 'label': '直平行流道'}
    }

    # 材料的线型样式
    materials_styles = {
        '材料1': {'linestyle': '-', 'label': '材料1'},
        '材料2': {'linestyle': '-', 'label': '材料2'},
        '材料3': {'linestyle': '-', 'label': '材料3'},
        '材料4': {'linestyle': '-', 'label': '材料4'},
        '材料5': {'linestyle': '-', 'label': '材料5'}
    }

    # 创建流道类型图例元素
    point_legend_elements = [plt.Line2D([0], [0],
                                        label=style['label'],
                                        marker=style['marker'],
                                        color=style['color'],
                                        linestyle='None',
                                        markersize=14)
                             for style in point_styles.values()]

    # 创建材料图例元素列表
    material_legend_elements = []
    material_labels = []

    # 存储所有材料的R²值和方程
    all_r2_texts = []

    # 第一轮：绘制图形并收集文本数据
    for idx, (material, data) in enumerate(all_results.items(), 1):
        # 将数据展开成x和y的形式
        all_data = data['all_data']
        x_data = np.concatenate([d[0] for d in all_data])

        y_data = np.concatenate([d[1] for d in all_data])
        # print(x_data,y_data)
        # 使用 LinearRegression 拟合直线
        model_lr = LinearRegression()
        model_lr.fit(x_data.reshape(-1, 1), y_data)

        # 获取拟合线的斜率和截距
        k = model_lr.coef_[0]
        b = model_lr.intercept_

        # 计算 R²
        r2 = model_lr.score(x_data.reshape(-1, 1), y_data)
        point_types = data['point_types']
        material_style = materials_styles[material]
        # 为每种点型分别画散点，使用对应的颜色
        for point_type, style in point_styles.items():
            mask = np.array(point_types) == point_type
            print(f"Point type: {point_type}, Mask sum: {np.sum(mask)}")  # 打印掩码的和，检查是否有匹配的点
            if np.any(mask):
                plt.scatter(x_data[mask], y_data[mask],
                            c=style['color'],
                            marker=style['marker'],
                            alpha=0.6,s=150)

        # 绘制拟合线，使用黑色但不同线型来区分材料
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = k * x_fit + b
        plt.plot(x_fit, y_fit,
                 color='black',
                 linestyle=material_style['linestyle'],
                 alpha=0.8,
                 linewidth=2)

        # 收集R²值和方程文本
        # 根据截距的正负选择连接符号
        # 生成R²值和方程文本
        equation_sign = " + " if b >= 0 else " - "
        equation = f"y = {k:.4f}x{equation_sign}{abs(b):.4f}"
        # equation = f"y5 = {k.item():.4f}x{equation_sign}{abs(b.item()):.4f}"
        all_r2_texts.append((material, f" R² = {r2:.4f}, {equation}"))

        material_legend_elements.append(
            plt.Line2D([0], [0],
                       color='black',
                       linestyle=material_style['linestyle'],
                       label=material_style['label'],
                       linewidth=2)
        )
        material_labels.append(material_style['label'])

    # 设置文本起始位置
    r2_y_position = 0.93
    equation_y_position = 0.95 - len(all_results) * 0.05

    # 添加背景框
    box_height = len(all_r2_texts) * 0.07
    plt.gca().add_patch(plt.Rectangle((0.01, r2_y_position - box_height + 0.06),
                                      0.81,  # 框的宽度
                                      box_height,  # 框的高度
                                      transform=plt.gca().transAxes,
                                      facecolor='white',
                                      edgecolor='black',
                                      alpha=0.7,
                                      linewidth=1))

    # 显示所有R²值
    for material, r2_text in all_r2_texts:
        plt.text(0.02, r2_y_position, r2_text,
                 transform=plt.gca().transAxes,
                 color='black',
                 fontsize=28)
        r2_y_position -= 0.05



    # 创建两个图例
    # 流道类型图例
    # point_legend = ax.legend(handles=point_legend_elements,
    #                          bbox_to_anchor=(0.31, 0.89),
    #                          loc='upper right',
    #                          frameon=True,
    #                          prop={'size': 23}
    #                          )



    # 添加第一个图例回图中
    # ax.add_artist(point_legend)

    plt.xlabel('特征值',fontsize=30,fontproperties=font)
    plt.ylabel('MEA均匀度',fontsize=30,fontproperties=font)

    plt.tick_params(axis='x', labelsize=25)  # 设置x轴刻度标签字体大小
    plt.tick_params(axis='y', labelsize=25)  # 设置y轴刻度标签字体大小
    # 调整布局以确保图例不被裁剪
    plt.tight_layout()
    plt.subplots_adjust(left=0.15,right=0.96)  # 为右侧图例留出空间

    plt.savefig(f'{type}_materials_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def five1(files):
    # try:
        # 设置中文字体
    font_path = '仿宋_GB2312.ttf'
    font = FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font.get_name()]
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    X,point_types = parse_excel_data1("data.xlsx")
    predict_and_save(X,"data.xlsx")


    all_results = {}

    # 处理每个文件
    for i, file_path in enumerate(files, 1):
        print(f"\n处理材料 {i}...")
        material_name = f'材料5'
        # 重置随机种子
        torch.manual_seed(42)
        np.random.seed(42)

        X,y,point_types = parse_excel_data(file_path)

        if len(X) == 0 or len(y) == 0:
            print(f"警告：{file_path} 没有加载到有效数据")
            continue
            # 为每个材料创建独立的StandardScaler
        scaler = StandardScaler()
        # 训练模型并获取预测结果
        all_data= train_and_evaluate_model(X, y,scaler)
        # 存储模型参数和结果
        all_results[material_name] = {
            'all_data': all_data,
            'point_types': point_types,
        }
    # 绘制所有材料的对比图
    plot_results_multiple(all_results,point_types[0])

# 按钮点击状态
def click_button():
    st.session_state.clicked = True

st.set_page_config(page_title="Predict")
st.markdown("# Predict")

# 选择输入样本数量（最少 5 个）
num_samples = st.number_input("选择输入样本数", min_value=5, max_value=10, value=5, step=1)

# 选择流道类型
option1 = st.selectbox("选择流道", ["蛇形流道", "直平行流道", "交指型流道", "网格型流道"])

# 动态生成多个输入框
input_data = []
for i in range(num_samples):
    st.markdown(f"### 样本 {i+1}")
    sample = {
        "通道类型":option1,
        "极板长度(nm)": st.text_input(f"极板长度(nm) - 样本 {i+1}"),
        "肋宽度(mm)": st.text_input(f"肋宽度(mm) - 样本 {i+1}"),
        "肋高度(mm)": st.text_input(f"肋高度(mm) - 样本 {i+1}"),
        "流道宽度(mm)": st.text_input(f"流道宽度(mm) - 样本 {i+1}"),
        "流道深度(mm)": st.text_input(f"流道深度(mm) - 样本 {i+1}"),
        "流道长度(mm)": st.text_input(f"流道长度(mm) - 样本 {i+1}"),
        "压力(kpa)": st.text_input(f"压力(kpa) - 样本 {i+1}"),
        "阳极H2O的质量分数": st.text_input(f"阳极H2O的质量分数 - 样本 {i+1}"),
        "阳极H2质量分数": st.text_input(f"阳极H2质量分数 - 样本 {i+1}"),
        "阴极H2O的质量分数": st.text_input(f"阴极H2O的质量分数 - 样本 {i+1}"),
        "阴极O2的质量分数": st.text_input(f"阴极O2的质量分数 - 样本 {i + 1}"),
        "阳极质量流量(kg/s)": st.text_input(f"阳极质量流量(kg/s) - 样本 {i+1}"),
        "阴极质量流量(kg/s)": st.text_input(f"阴极质量流量(kg/s) - 样本 {i+1}"),
        "开路电压(V)": st.text_input(f"开路电压(V) - 样本 {i+1}"),
        "温度(℃)": st.text_input(f"温度(℃) - 样本 {i+1}"),
        "功率密度(w/cm2)":st.text_input(f"功率密度(w/cm2) - 样本{i+1}"),
        "Predict":""
    }
    input_data.append(sample)

# 预测按钮
st.button("Predict", on_click=click_button)

# 存储点击状态
if "clicked" not in st.session_state:
    st.session_state.clicked = False
files = [
     "data.xlsx"
]
# 处理点击事件
if st.session_state.clicked:
    df = pd.DataFrame(input_data)
    df.to_excel("data.xlsx", index=False, engine='openpyxl')
    five1(files)
    time.sleep(1)
    map2( "data.xlsx",option1)
    try:
        # 打开图片
        image1 = Image.open(f"{option1}_materials_comparison.png")
        image2 = Image.open(f"{option1}_materials_comparison1.png")
        # 显示图片
        st.image(image1, caption='处理后的图片', use_column_width=True)
        st.image(image2, caption='处理后的图片', use_column_width=True)
    except Exception as e:
        st.error(f"无法打开图片：{e}")