import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def process_data_group(data, group_type):
    """处理单组数据"""
    X = []
    y = []
    point_types = []
    try:
        # 数据起始列（第6列，索引为5）
        START_COL = 5
        SAMPLES_PER_GROUP = 9  # 每组9个样本

        # 找到各场的起始行
        concentration_start = data[data.iloc[:, 0] == '浓度场均匀性系数'].index[0] + 1
        velocity_start = data[data.iloc[:, 0] == '速度场均匀性系数'].index[0] + 1
        pressure_start = data[data.iloc[:, 0] == '压力场均匀性系数'].index[0] + 1
        temperature_start = data[data.iloc[:, 0] == '温度场均匀性系数'].index[0] + 1
        power_density_row = data[data.iloc[:, 0] == '阴极功率密度'].index[0]

        # 从第6列开始处理数据
        for col in range(START_COL, START_COL + SAMPLES_PER_GROUP):
            if col >= len(data.columns):  # 防止列索引超出范围
                break

            depth = data.columns[col]  # 获取深度值

            features = []

            # 收集浓度场的8个值
            concentration_values = data.iloc[concentration_start:concentration_start + 6, col].values
            features.extend(concentration_values.astype(float))

            # 收集速度场的8个值
            velocity_values = data.iloc[velocity_start:velocity_start + 6, col].values
            features.extend(velocity_values.astype(float))

            # 收集压力场的8个值
            pressure_values = data.iloc[pressure_start:pressure_start + 6, col].values
            features.extend(pressure_values.astype(float))

            # 收集温度场的8个值
            temperature_values = data.iloc[temperature_start:temperature_start + 6, col].values
            features.extend(temperature_values.astype(float))

            # 验证特征数量
            if len(features) != 24:
                print(f"警告：深度 {depth} 的特征数量为{len(features)}，期望值为24")
                continue

            # 获取输出值（阴极功率密度）
            power_density = float(data.iloc[power_density_row, col])
            if power_density is not None and not np.isnan(power_density):
                X.append(features)
                y.append(power_density)
                point_types.append(group_type)

        X = np.array(X)
        y = np.array(y)

        return X, y, point_types

    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        return np.array([]), np.array([]), []


def get_point_type(group_type, first_column_header):
    if '网格' in first_column_header:
        group_type = '网格型'
    elif '蛇形' in first_column_header:
        group_type = '蛇形'
    elif '直平行' in first_column_header:
        group_type = '直平行'
    elif '交指' in first_column_header:
        group_type = '交指型'
    # 如果没有任何匹配，保持原来的 group_type 不变

    return group_type


def process_all_groups(file_path):
    """处理所有组的数据"""
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 找到所有组的起始行
        group_starts = df[df.iloc[:, 0].str.contains('流道宽', na=False)].index

        all_X = []
        all_y = []
        all_types = []
        group_type = ""

        # 处理每一组数据
        for i in range(len(group_starts)):
            start = group_starts[i]

            if i == 0:
                first_column_header = df.columns[0]
                group_type = get_point_type(group_type, first_column_header)
            else:
                first_column_header = df.iloc[start - 1, 0]
                group_type = get_point_type(group_type, first_column_header)

            end = group_starts[i + 1] if i < len(group_starts) - 1 else len(df)

            group_data = df.iloc[start:end].reset_index(drop=True)
            print(f"\n处理第 {i + 1} 组数据...")

            X, y, types = process_data_group(group_data, group_type)
            all_X.extend(X)
            all_y.extend(y)
            all_types.extend(types)

        all_X = np.array(all_X)
        all_y = np.array(all_y)

        print(f"\n所有数据处理完成:")
        print(f"总样本数量: {len(all_X)} (预期应为: {len(group_starts) * 9})")
        print(f"输入特征形状: {all_X.shape}")
        print(f"输出值形状: {all_y.shape}")

        return all_X, all_y, all_types

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return np.array([]), np.array([]), []


def train_and_evaluate_model(X, y):
    """训练和评估模型"""
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练模型
    model = LinearRegression()
    model.fit(X_scaled, y)

    # 预测
    y_pred = model.predict(X_scaled)

    # 计算R²值
    r2 = r2_score(y, y_pred)

    return model, scaler, X, y, y_pred, r2


def plot_results_multiple(all_results):
    """绘制多个材料的散点和拟合直线"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # 流道类型的颜色样式
    point_styles = {
        '网格型': {'color': 'red', 'marker': 'o', 'label': '网格型流道'},
        '蛇形': {'color': 'blue', 'marker': 's', 'label': '蛇形流道'},
        '交指型': {'color': 'green', 'marker': 'x', 'label': '交指型流道'},
        '直平行': {'color': 'purple', 'marker': '*', 'label': '直平行流道'}
    }

    # 材料的线型样式
    materials_styles = {
        '材料1': {'linestyle': '-', 'label': '材料1'},
        '材料2': {'linestyle': '--', 'label': '材料2'},
        '材料3': {'linestyle': ':', 'label': '材料3'},
        '材料4': {'linestyle': '-.', 'label': '材料4'},
        '材料5': {'linestyle': '--', 'label': '材料5'}
    }

    # 创建流道类型图例元素
    point_legend_elements = [plt.Line2D([0], [0],
                                        label=style['label'],
                                        marker=style['marker'],
                                        color=style['color'],
                                        linestyle='None',
                                        markersize=12)
                             for style in point_styles.values()]

    # 创建材料图例元素列表
    material_legend_elements = []
    material_labels = []

    # 存储所有材料的R²值和方程
    all_r2_texts = []

    # 第一轮：绘制图形并收集文本数据
    for idx, (material, data) in enumerate(all_results.items(), 1):
        y_true = data['y_true']
        y_predict = data['y_pred']
        r2 = data['r2']
        X = data['X']
        point_types = data['point_types']
        material_style = materials_styles[material]

        # 计算32维特征的平均值作为X轴
        X_mean = np.mean(X, axis=1)

        # 为每种点型分别画散点，使用对应的颜色
        for point_type, style in point_styles.items():
            mask = np.array(point_types) == point_type
            if np.any(mask):
                plt.scatter(X_mean[mask], y_true[mask],
                            c=style['color'],
                            marker=style['marker'],
                            alpha=0.6,s=150)

        # 绘制拟合线，使用黑色但不同线型来区分材料
        x_line = np.linspace(X_mean.min(), X_mean.max(), 100)
        z_pred = np.polyfit(X_mean, y_predict, 1)
        p_pred = np.poly1d(z_pred)
        plt.plot(x_line, p_pred(x_line),
                 color='black',
                 linestyle=material_style['linestyle'],
                 alpha=0.8,
                 linewidth=2)

        # 收集R²值和方程文本
        # 根据截距的正负选择连接符号
        intercept_sign = " + " if z_pred[1] >= 0 else " "
        equation = f"y{5} = {z_pred[0]:.4f}x{intercept_sign}{z_pred[1]:.4f}"
        all_r2_texts.append((material, f"y{5} = {z_pred[0]:.4f}x{intercept_sign}{z_pred[1]:.4f}  R² = {r2:.4f}"))



        material_legend_elements.append(
            plt.Line2D([0], [0],
                       color='black',
                       linestyle=material_style['linestyle'],
                       label=material_style['label'],
                       linewidth=2)
        )
        material_labels.append(material_style['label'])

    # 设置文本起始位置
    r2_y_position = 0.92
    equation_y_position = 0.95 - len(all_results) * 0.05

    # 添加背景框
    box_height = len(all_r2_texts) * 0.07
    plt.gca().add_patch(plt.Rectangle((0.01, r2_y_position - box_height + 0.06),
                                      0.68,  # 框的宽度
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
                 fontsize=30)
        r2_y_position -= 0.06



    # 创建两个图例
    # 流道类型图例
    point_legend = ax.legend(handles=point_legend_elements,
                             bbox_to_anchor=(0.25, 0.89),
                             loc='upper right',
                             frameon=True,
                             )
    for text in point_legend.get_texts():
        text.set_fontsize(25)


    # 添加第一个图例回图中
    ax.add_artist(point_legend)

    plt.xlabel('MEA均匀度',fontsize=30)
    plt.ylabel('输出功率密度/(W/cm2)',fontsize=30)

    # plt.grid(True, alpha=0.3)
    plt.tick_params(axis='x', labelsize=25)  # 设置x轴刻度标签字体大小
    plt.tick_params(axis='y', labelsize=25)  # 设置y轴刻度标签字体大小
    # 调整布局以确保图例不被裁剪
    plt.tight_layout()
    plt.subplots_adjust(left=0.12,right=0.96)  # 为右侧图例留出空间

    plt.savefig('all_materials_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    try:
        # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

        # 文件列表
        files = [
            #"第一种材料.xlsx",
            # "第二种材料.xlsx",
            # "第三种材料.xlsx",
            # "第四种材料.xlsx",
             "data.xlsx"
        ]

        all_results = {}

        # 处理每个文件
        for i, file_path in enumerate(files, 1):
            print(f"\n处理材料 {i}...")
            material_name = f'材料{i}'

            X, y, point_types = process_all_groups(file_path)

            if len(X) == 0 or len(y) == 0:
                print(f"警告：{file_path} 没有加载到有效数据")
                continue

            # 训练模型并获取预测结果
            model, scaler, X, y, y_pred, r2 = train_and_evaluate_model(X, y)

            # 存储模型参数和结果
            all_results[material_name] = {
                'y_true': y,
                'y_pred': y_pred,
                'r2': r2,
                'point_types': point_types,
                'X': X,
                'model': model,
                'scaler': scaler
            }

        # 绘制所有材料的对比图
        plot_results_multiple(all_results)

    except Exception as e:
        print(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()
