import time
import five2
import streamlit as st
import pandas as pd
from PIL import Image
import five1


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
    five1.five1(files)
    time.sleep(1)
    five2.map2( "data.xlsx",option1)
    try:
        # 打开图片
        image1 = Image.open(f"{option1}_materials_comparison.png")
        image2 = Image.open(f"{option1}_materials_comparison1.png")
        # 显示图片
        st.image(image1, caption='处理后的图片', use_column_width=True)
        st.image(image2, caption='处理后的图片', use_column_width=True)
    except Exception as e:
        st.error(f"无法打开图片：{e}")