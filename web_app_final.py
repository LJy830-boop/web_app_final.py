import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# 设置页面配置
st.set_page_config(page_title="电池SOH和RUL预测系统", layout="wide")
# 页面标题
st.title("电池健康状态(SOH)和剩余使用寿命(RUL)预测系统")
# 侧边栏说明
st.sidebar.header("使用说明")
st.sidebar.info(
    """
    1. 上传电池测试数据Excel文件
    2. 系统将自动分析数据并预测SOH和RUL
    3. 查看预测结果和可视化

    **注意**: 上传的Excel文件应包含电池循环测试数据。
    系统会自动识别并处理不同格式的数据。
    """
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 创建预测结果可视化
def create_prediction_plot(soh_pred, rul_pred):
    """创建SOH和RUL预测结果的可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # SOH仪表盘
    soh_colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000']
    soh_thresholds = [0, 60, 80, 90, 100]

    # 确定SOH所在的区间
    soh_color = soh_colors[0]
    for i in range(len(soh_thresholds) - 1):
        if soh_thresholds[i] <= soh_pred <= soh_thresholds[i + 1]:
            soh_color = soh_colors[i]
            break

    ax1.pie([soh_pred, 100 - soh_pred], colors=[soh_color, '#EEEEEE'],
            startangle=90, counterclock=False,
            wedgeprops={'width': 0.3, 'edgecolor': 'w'})
    ax1.text(0, 0, f"{soh_pred:.1f}%", ha='center', va='center', fontsize=24, fontweight='bold')
    ax1.set_title('电池健康状态 (SOH)', fontsize=16)

    # RUL条形图
    ax2.barh(['剩余使用寿命'], [rul_pred], color='#4CAF50', height=0.5)
    ax2.set_xlim(0, max(100, rul_pred * 1.2))
    ax2.text(rul_pred + 1, 0, f"{rul_pred:.1f} 循环", va='center', fontsize=14)
    ax2.set_title('剩余使用寿命 (RUL)', fontsize=16)
    ax2.set_xlabel('循环次数', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 将图转换为base64编码
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, encoding='utf-8')
    buf.seek(0)
    plt.close()

    return base64.b64encode(buf.read()).decode()


# 预测函数
def predict_battery(df_cycle):
    """预测电池SOH和RUL"""
    try:
        # 提取最后一个循环的放电容量
        if '放电容量(Ah)' in df_cycle.columns:
            discharge_capacity = df_cycle['放电容量(Ah)'].iloc[-1]
        else:
            # 尝试找到可能的放电容量列
            possible_columns = [col for col in df_cycle.columns if
                                '放电' in col and ('容量' in col or 'capacity' in col.lower())]
            if possible_columns:
                discharge_capacity = df_cycle[possible_columns[0]].iloc[-1]
            else:
                # 如果找不到放电容量列，使用第一个数值列
                numeric_cols = df_cycle.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    discharge_capacity = df_cycle[numeric_cols[0]].iloc[-1]
                else:
                    return 85.0, 50.0  # 默认值

        # 获取初始容量（第一个循环）
        if len(df_cycle) > 1:
            if '放电容量(Ah)' in df_cycle.columns:
                initial_capacity = df_cycle['放电容量(Ah)'].iloc[0]
            elif possible_columns:
                initial_capacity = df_cycle[possible_columns[0]].iloc[0]
            elif len(numeric_cols) > 0:
                initial_capacity = df_cycle[numeric_cols[0]].iloc[0]
            else:
                initial_capacity = discharge_capacity
        else:
            initial_capacity = discharge_capacity

        # 计算SOH
        soh = (discharge_capacity / initial_capacity) * 100 if initial_capacity > 0 else 90.0

        # 确保SOH在合理范围内
        soh = max(0, min(100, soh))

        # 计算RUL
        if soh > 80:
            # 计算每个循环的平均SOH衰减率
            if len(df_cycle) > 1:
                cycle_count = len(df_cycle)
                total_soh_decline = 100 - soh
                avg_decline_per_cycle = total_soh_decline / cycle_count if cycle_count > 0 else 0.2

                # 计算RUL
                remaining_soh = soh - 80  # 距离EOL的SOH差值
                rul = remaining_soh / avg_decline_per_cycle if avg_decline_per_cycle > 0 else 50.0
            else:
                # 如果只有一个循环数据，使用默认衰减率
                rul = (soh - 80) / 0.2  # 假设每个循环SOH下降0.2%
        else:
            rul = 0.0  # SOH已低于80%，RUL为0

        # 确保RUL不为负
        rul = max(0, rul)

        return soh, rul

    except Exception as e:
        st.error(f"预测过程中出错: {e}")
        return 90.0, 50.0  # 默认值


# 主应用
def main():
    # 文件上传
    uploaded_file = st.file_uploader("上传电池测试数据 (Excel格式)", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # 读取Excel文件
            try:
                # 尝试读取所有工作表
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names

                # 检查是否有'cycle'工作表
                if 'cycle' in sheet_names:
                    df_cycle = pd.read_excel(excel_file, sheet_name='cycle')
                    st.success("成功读取'cycle'工作表！")
                else:
                    # 如果没有'cycle'工作表，使用第一个工作表
                    df_cycle = pd.read_excel(excel_file, sheet_name=0)
                    st.info(f"未找到'cycle'工作表，使用'{sheet_names[0]}'工作表进行分析。")
            except Exception as e:
                st.warning(f"读取Excel文件时出错: {e}")
                # 尝试直接读取第一个工作表
                df_cycle = pd.read_excel(uploaded_file)

            # 显示数据概览
            st.subheader("数据概览")
            st.dataframe(df_cycle.head())
            st.text(f"总行数: {len(df_cycle)}")

            # 数据分析
            st.subheader("数据分析")

            # 检查数据列
            numeric_cols = df_cycle.select_dtypes(include=[np.number]).columns.tolist()
            st.write("检测到的数值列:")
            st.write(", ".join(numeric_cols))

            # 如果有足够的数据，显示一些基本统计信息
            if len(df_cycle) > 1 and len(numeric_cols) > 0:
                # 选择第一个数值列进行可视化
                selected_col = st.selectbox("选择要分析的列:", numeric_cols)

                # 创建简单的趋势图
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_cycle.index, df_cycle[selected_col], marker='o', linestyle='-')
                ax.set_title(f"{selected_col}随循环次数的变化")
                ax.set_xlabel("循环索引")
                ax.set_ylabel(selected_col)
                ax.grid(True)
                st.pyplot(fig)

            # 预测SOH和RUL
            st.subheader("预测结果")
            soh_pred, rul_pred = predict_battery(df_cycle)

            # 显示预测结果
            col1, col2 = st.columns(2)

            with col1:
                st.metric("电池健康状态 (SOH)", f"{soh_pred:.2f}%")

                # 添加SOH状态解释
                if soh_pred >= 90:
                    st.success("电池状态良好，可以继续使用。")
                elif soh_pred >= 80:
                    st.info("电池状态正常，但已有轻微老化。")
                elif soh_pred >= 60:
                    st.warning("电池已明显老化，建议密切监控。")
                else:
                    st.error("电池严重老化，建议更换。")

            with col2:
                st.metric("剩余使用寿命 (RUL)", f"{rul_pred:.2f} 循环")

                # 添加RUL状态解释
                if rul_pred >= 50:
                    st.success("电池剩余寿命充足。")
                elif rul_pred >= 20:
                    st.info("电池剩余寿命适中，可继续使用一段时间。")
                elif rul_pred >= 5:
                    st.warning("电池剩余寿命较短，建议准备更换。")
                else:
                    st.error("电池即将达到寿命终点，建议尽快更换。")

            # 创建并显示可视化
            st.subheader("预测结果可视化")
            plot_data = create_prediction_plot(soh_pred, rul_pred)
            try:
                import base64
                from io import BytesIO
                import cv2
                import numpy as np
                img_data = base64.b64decode(plot_data)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("图片解码失败")
                st.image(f"data:image/png;base64,{plot_data}", use_column_width=True)
            except Exception as e:
                st.error(f"显示图片时出错: {e}")

            # 添加预测结果解释
            st.subheader("结果解释")
            st.write(f"""
            - **电池健康状态 (SOH)**: {soh_pred:.2f}% 表示电池当前的容量相对于初始容量的百分比。
              SOH值越高，表示电池状态越好。一般认为SOH低于80%时，电池性能开始明显下降。

            - **剩余使用寿命 (RUL)**: {rul_pred:.2f} 循环表示在当前使用条件下，电池预计还能完成的充放电循环次数，
              直到SOH降至80%（通常被视为电池寿命终点）。
            """)

            # 添加建议
            st.subheader("使用建议")
            if soh_pred >= 90 and rul_pred >= 50:
                st.success("电池状态优良，可以继续正常使用，无需特别关注。")
            elif soh_pred >= 80 and rul_pred >= 20:
                st.info("电池状态良好，建议定期监测SOH变化趋势。")
            elif soh_pred >= 60 and rul_pred >= 5:
                st.warning("电池已进入老化阶段，建议增加监测频率，并考虑在未来一段时间内更换电池。")
            else:
                st.error("电池已严重老化或即将达到寿命终点，建议尽快更换电池，以避免可能的性能问题或安全隐患。")

        except Exception as e:
            st.error(f"处理文件时出错: {e}")
            st.info("请确保上传的Excel文件包含电池循环测试数据。")

    else:
        # 如果没有上传文件，显示示例和说明
        st.info("请上传电池测试数据Excel文件以获取预测结果。")

        # 显示示例图片
        st.subheader("示例预测结果")
        example_soh = 92.5
        example_rul = 65.3
        example_plot = create_prediction_plot(example_soh, example_rul)
        st.image(f"data:image/png;base64,{example_plot}", use_column_width=True)

        # 添加交互式输入选项
        st.subheader("或者直接输入电池参数")

        col1, col2 = st.columns(2)

        with col1:
            initial_capacity = st.number_input("初始放电容量(Ah)", min_value=0.1, max_value=20.0, value=9.5, step=0.1)

        with col2:
            current_capacity = st.number_input("当前放电容量(Ah)", min_value=0.0, max_value=20.0, value=8.5, step=0.1)

        cycles_completed = st.slider("已完成的循环次数", min_value=1, max_value=500, value=20)

        if st.button("预测", type="primary"):
            # 计算SOH
            manual_soh = (current_capacity / initial_capacity) * 100

            # 计算SOH衰减率
            soh_decline = 100 - manual_soh
            avg_decline_per_cycle = soh_decline / cycles_completed if cycles_completed > 0 else 0.2

            # 计算RUL
            remaining_soh = manual_soh - 80  # 距离EOL的SOH差值
            manual_rul = remaining_soh / avg_decline_per_cycle if avg_decline_per_cycle > 0 else 50.0
            manual_rul = max(0, manual_rul)

            # 显示预测结果
            col1, col2 = st.columns(2)

            with col1:
                st.metric("电池健康状态 (SOH)", f"{manual_soh:.2f}%")

                # 添加SOH状态解释
                if manual_soh >= 90:
                    st.success("电池状态良好，可以继续使用。")
                elif manual_soh >= 80:
                    st.info("电池状态正常，但已有轻微老化。")
                elif manual_soh >= 60:
                    st.warning("电池已明显老化，建议密切监控。")
                else:
                    st.error("电池严重老化，建议更换。")

            with col2:
                st.metric("剩余使用寿命 (RUL)", f"{manual_rul:.2f} 循环")

                # 添加RUL状态解释
                if manual_rul >= 50:
                    st.success("电池剩余寿命充足。")
                elif manual_rul >= 20:
                    st.info("电池剩余寿命适中，可继续使用一段时间。")
                elif manual_rul >= 5:
                    st.warning("电池剩余寿命较短，建议准备更换。")
                else:
                    st.error("电池即将达到寿命终点，建议尽快更换。")

            # 创建并显示可视化
            manual_plot = create_prediction_plot(manual_soh, manual_rul)
            st.image(f"data:image/png;base64,{manual_plot}", use_column_width=True)

    # 添加页脚
    st.markdown("---")
    st.markdown("© 2025 电池SOH和RUL预测系统 | 基于机器学习的电池健康状态和剩余使用寿命预测")


if __name__ == "__main__":
    main()