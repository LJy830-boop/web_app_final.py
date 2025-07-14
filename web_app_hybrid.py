import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 避免负号显示异常
plt.rcParams['axes.unicode_minus'] = False

# 设置页面配置
st.set_page_config(page_title="电池SOH和RUL预测系统", layout="wide")

# 使用markdown避免字符显示问题
st.markdown("# 电池健康状态(SOH)和剩余使用寿命(RUL)预测系统 ")

# 侧边栏说明
st.sidebar.markdown("### 使用说明")
st.sidebar.info(
    """
    1. 上传电池测试数据Excel文件
    2. 系统将自动分析数据并预测SOH和RUL
    3. 查看预测结果和可视化

    **注意**: 上传的Excel文件应包含电池循环测试数据。
    系统会自动识别并处理不同格式的数据。
    """
)

# 创建预测结果可视化
def create_prediction_plot(soh_pred, rul_pred):
    """创建SOH和RUL预测结果的可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # SOH仪表盘
    soh_colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000']
    soh_thresholds = [0, 60, 80, 90, 100]

    # 确定SOH所在的区间
    soh_color = soh_colors[0]
    for i in range(len(soh_thresholds)-1):
        if soh_thresholds[i] <= soh_pred <= soh_thresholds[i+1]:
            soh_color = soh_colors[i]
            break

    ax1.pie([soh_pred, 100-soh_pred], colors=[soh_color, '#EEEEEE'], 
            startangle=90, counterclock=False, 
            wedgeprops={'width': 0.3, 'edgecolor': 'w'})
    ax1.text(0, 0, f"{soh_pred:.1f}%", ha='center', va='center', fontsize=24, fontweight='bold')
    ax1.set_title('电池健康状态 (SOH)', fontsize=16)

    # RUL条形图
    ax2.barh(['剩余使用寿命'], [rul_pred], color='#4CAF50', height=0.5)
    ax2.set_xlim(0, max(100, rul_pred*1.2))
    ax2.text(rul_pred+1, 0, f"{rul_pred:.1f} 循环", va='center', fontsize=14)
    ax2.set_title('剩余使用寿命 (RUL)', fontsize=16)
    ax2.set_xlabel('循环次数', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    
    # 将图转换为base64编码
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.read()).decode()

# 预测函数 - 混合版本（保持SOH低于80%时RUL为0，但加入其他增强功能）
def predict_battery(df_cycle, use_nonlinear_model=True, expected_total_cycles=500):
    """预测电池SOH和RUL - 混合版本，保持SOH低于80%时RUL为0，但加入其他增强功能"""
    try:
        # 提取最后一个循环的放电容量
        if '放电容量(Ah)' in df_cycle.columns:
            discharge_capacity = df_cycle['放电容量(Ah)'].iloc[-1]
        else:
            # 尝试找到可能的放电容量列
            possible_columns = [col for col in df_cycle.columns if '放电' in col and ('容量' in col or 'capacity' in col.lower())]
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
        
        # 增强的RUL计算 - 但保持SOH低于80%时RUL为0
        cycle_count = len(df_cycle)
        
        # 如果SOH已经低于80%，直接返回RUL=0
        if soh <= 80:
            return soh, 0.0
        
        # 如果有足够的数据点，尝试使用非线性衰减模型
        if use_nonlinear_model and cycle_count >= 5 and '放电容量(Ah)' in df_cycle.columns:
            # 提取所有循环的容量数据
            capacities = df_cycle['放电容量(Ah)'].values
            cycles = np.arange(len(capacities))
            
            # 计算最近的衰减率（使用最后30%的数据或至少3个点）
            recent_points = max(3, int(cycle_count * 0.3))
            recent_capacities = capacities[-recent_points:]
            recent_cycles = cycles[-recent_points:]
            
            if len(recent_capacities) > 1:
                # 计算最近的衰减率
                recent_decline = (recent_capacities[0] - recent_capacities[-1]) / len(recent_capacities)
                recent_decline_percent = (recent_decline / initial_capacity) * 100
                
                # 应用加速因子 - 随着循环次数增加，衰减会加速
                acceleration_factor = 1.0 + (cycle_count / 200)  # 随着循环次数增加，加速因子增大
                future_decline_percent = recent_decline_percent * acceleration_factor
                
                # 计算RUL - 只计算达到80%SOH还需要的循环次数
                remaining_soh = soh - 80
                rul = remaining_soh / future_decline_percent if future_decline_percent > 0 else 50.0
                
                # 设置合理上限 - 基于电池类型和当前循环数
                remaining_cycles = expected_total_cycles - cycle_count
                rul = min(rul, remaining_cycles)
                
                # 确保RUL不为负且有合理上限
                rul = max(0, min(rul, 200))  # 设置最大RUL为200循环
                
                return soh, rul
        
        # 如果没有足够数据或上面的方法失败，使用简化方法
        # 计算平均SOH衰减率
        if cycle_count > 1:
            total_soh_decline = 100 - soh
            avg_decline_per_cycle = total_soh_decline / cycle_count if cycle_count > 0 else 0.2
            
            # 应用加速因子
            if use_nonlinear_model:
                acceleration_factor = 1.0 + (cycle_count / 200)
                future_decline_per_cycle = avg_decline_per_cycle * acceleration_factor
            else:
                future_decline_per_cycle = avg_decline_per_cycle
            
            # 计算RUL - 只计算达到80%SOH还需要的循环次数
            remaining_soh = soh - 80
            rul = remaining_soh / future_decline_per_cycle if future_decline_per_cycle > 0 else 50.0
            
            # 设置合理上限
            remaining_cycles = expected_total_cycles - cycle_count
            rul = min(rul, remaining_cycles)
            
            # 确保RUL不为负且有合理上限
            rul = max(0, min(rul, 200))
        else:
            # 如果只有一个循环数据，使用默认值
            rul = 50.0
        
        return soh, rul
    
    except Exception as e:
        st.error(f"预测过程中出错: {e}")
        return 90.0, 50.0  # 默认值

# 主应用
def main():
    # 文件上传
    uploaded_file = st.file_uploader("上传电池测试数据 (Excel格式)", type=["xlsx", "xls"])
    
    # 添加高级选项
    with st.expander("高级选项"):
        use_nonlinear_model = st.checkbox("启用非线性衰减模型", value=True, 
                                         help="考虑电池在生命周期后期加速衰减的特性")
        expected_total_cycles = st.slider("预期总循环寿命", min_value=100, max_value=1000, value=500,
                                         help="设置电池预期的总循环寿命，用于限制RUL预测的上限")
        st.info("注意：无论使用何种模型，当SOH低于80%时，RUL将始终为0，表示电池已达到寿命终点。")
    
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
            st.markdown("## 数据概览")
            st.dataframe(df_cycle.head())
            st.text(f"总行数: {len(df_cycle)}")
            
            # 数据分析
            st.markdown("## 数据分析")
            
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
            st.markdown("## 预测结果")
            soh_pred, rul_pred = predict_battery(df_cycle, use_nonlinear_model, expected_total_cycles)
            
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
                    st.error("电池严重老化，建议尽快更换。")
            
            with col2:
                st.metric("剩余使用寿命 (RUL)", f"{rul_pred:.2f} 循环")
                
                # 添加RUL状态解释 - 混合版本
                if rul_pred > 50:
                    st.success("电池剩余寿命充足。")
                elif rul_pred > 20:
                    st.info("电池剩余寿命适中，可继续使用一段时间。")
                elif rul_pred > 0:
                    st.warning("电池剩余寿命较短，建议准备更换。")
                else:
                    st.error("电池已达到寿命终点，建议尽快更换。")
            
            # 创建并显示可视化
            st.markdown("## 预测结果可视化")
            plot_data = create_prediction_plot(soh_pred, rul_pred)
            st.image(f"data:image/png;base64,{plot_data}", use_column_width=True)
            
            # 添加预测结果解释 - 混合版本
            st.markdown("## 结果解释")
            st.write(f"""
            - **电池健康状态 (SOH)**: {soh_pred:.2f}% 表示电池当前的容量相对于初始容量的百分比。
              SOH值越高，表示电池状态越好。一般认为SOH低于80%时，电池性能开始明显下降。
            
            - **剩余使用寿命 (RUL)**: {rul_pred:.2f} 循环表示在当前使用条件下，电池预计还能完成的充放电循环次数。
              
              RUL计算基于以下标准：
              * 当SOH > 80%时：使用增强算法计算达到80%SOH还需要的循环次数
              * 当SOH ≤ 80%时：RUL为0，表示电池已达到寿命终点
              
              增强算法考虑了以下因素：
              * {'非线性衰减：电池在生命周期后期通常会加速衰减' if use_nonlinear_model else '线性衰减：假设电池以恒定速率衰减'}
              * 最近趋势：优先考虑最近的衰减数据
              * 合理上限：基于预期总循环寿命({expected_total_cycles}循环)设置上限
            """)
            
            # 添加建议 - 混合版本
            st.markdown("## 使用建议")
            if soh_pred >= 90 and rul_pred >= 50:
                st.success("电池状态优良，可以继续正常使用，无需特别关注。")
            elif soh_pred >= 80 and rul_pred >= 20:
                st.info("电池状态良好，建议定期监测SOH变化趋势。")
            elif soh_pred >= 80 and rul_pred > 0:
                st.warning("电池状态尚可，但剩余寿命较短，建议准备更换电池。")
            else:
                st.error("电池已达到寿命终点，建议尽快更换电池，以避免可能的性能问题或安全隐患。")
            
            # 添加详细分析 - 增强功能
            if len(df_cycle) > 5 and '放电容量(Ah)' in df_cycle.columns:
                st.markdown("## 详细分析")
                
                # 容量衰减趋势分析
                capacities = df_cycle['放电容量(Ah)'].values
                cycles = np.arange(len(capacities))
                
                # 计算衰减率
                if len(capacities) > 1:
                    total_decline = capacities[0] - capacities[-1]
                    avg_decline_per_cycle = total_decline / (len(capacities) - 1)
                    
                    # 计算最近的衰减率
                    recent_points = max(3, int(len(capacities) * 0.3))
                    recent_capacities = capacities[-recent_points:]
                    recent_decline = (recent_capacities[0] - recent_capacities[-1]) / len(recent_capacities)
                    
                    # 创建容量衰减趋势图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(cycles, capacities, marker='o', linestyle='-', label='实际容量')
                    
                    # 如果SOH > 80%，预测未来趋势
                    if soh_pred > 80:
                        # 预测未来趋势
                        future_cycles = np.arange(len(capacities), len(capacities) + int(rul_pred) + 10)
                        
                        if use_nonlinear_model:
                            # 非线性预测
                            acceleration_factor = 1.0 + (len(capacities) / 200)
                            future_decline = recent_decline * acceleration_factor
                            future_capacities = [capacities[-1]]
                            
                            for i in range(1, len(future_cycles)):
                                next_capacity = future_capacities[-1] - future_decline
                                future_capacities.append(max(0, next_capacity))
                            
                            ax.plot(future_cycles, future_capacities, linestyle='--', color='red', label='预测趋势(非线性)')
                        else:
                            # 线性预测
                            future_capacities = [capacities[-1] - avg_decline_per_cycle * i for i in range(1, len(future_cycles) + 1)]
                            ax.plot(future_cycles, future_capacities, linestyle='--', color='green', label='预测趋势(线性)')
                        
                        # 标记80% SOH点
                        eol_capacity = initial_capacity * 0.8
                        ax.axhline(y=eol_capacity, color='r', linestyle='-', alpha=0.5, label='80% SOH (寿命终点)')
                    
                    ax.set_title('电池容量衰减趋势分析')
                    ax.set_xlabel('循环次数')
                    ax.set_ylabel('放电容量 (Ah)')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    
                    # 显示衰减率信息
                    st.markdown("### 容量衰减率分析")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("初始容量", f"{capacities[0]:.4f} Ah")
                    with col2:
                        st.metric("当前容量", f"{capacities[-1]:.4f} Ah")
                    with col3:
                        st.metric("总衰减量", f"{total_decline:.4f} Ah")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("平均衰减率", f"{avg_decline_per_cycle:.6f} Ah/循环")
                    with col2:
                        st.metric("最近衰减率", f"{recent_decline:.6f} Ah/循环")
                    with col3:
                        if use_nonlinear_model:
                            acceleration_factor = 1.0 + (len(capacities) / 200)
                            st.metric("加速因子", f"{acceleration_factor:.2f}")
        
        except Exception as e:
            st.error(f"处理文件时出错: {e}")
            st.info("请确保上传的Excel文件包含电池循环测试数据。")
    
    else:
        # 如果没有上传文件，显示示例和说明
        st.info("请上传电池测试数据Excel文件以获取预测结果。")
        
        # 显示示例图片
        st.markdown("## 示例预测结果")
        example_soh = 92.5
        example_rul = 65.3
        example_plot = create_prediction_plot(example_soh, example_rul)
        st.image(f"data:image/png;base64,{example_plot}", use_column_width=True)
        
        # 添加交互式输入选项
        st.markdown("## 或者直接输入电池参数")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_capacity = st.number_input("初始放电容量(Ah)", min_value=0.1, max_value=20.0, value=9.5, step=0.1)
        
        with col2:
            current_capacity = st.number_input("当前放电容量(Ah)", min_value=0.0, max_value=20.0, value=8.5, step=0.1)
        
        cycles_completed = st.slider("已完成的循环次数", min_value=1, max_value=500, value=20)
        
        # 使用与高级选项相同的设置
        use_nonlinear_model_manual = use_nonlinear_model
        expected_total_cycles_manual = expected_total_cycles
        
        if st.button("预测", type="primary"):
            # 计算SOH
            manual_soh = (current_capacity / initial_capacity) * 100
            
            # 如果SOH低于80%，RUL直接为0
            if manual_soh <= 80:
                manual_rul = 0.0
            else:
                # 计算SOH衰减率
                soh_decline = 100 - manual_soh
                avg_decline_per_cycle = soh_decline / cycles_completed if cycles_completed > 0 else 0.2
                
                # 应用增强功能
                if use_nonlinear_model_manual:
                    # 应用加速因子
                    acceleration_factor = 1.0 + (cycles_completed / 200)
                    future_decline_per_cycle = avg_decline_per_cycle * acceleration_factor
                else:
                    future_decline_per_cycle = avg_decline_per_cycle
                
                # 计算RUL - 只计算达到80%SOH还需要的循环次数
                remaining_soh = manual_soh - 80
                manual_rul = remaining_soh / future_decline_per_cycle if future_decline_per_cycle > 0 else 50.0
                
                # 设置合理上限
                remaining_cycles = expected_total_cycles_manual - cycles_completed
                manual_rul = min(manual_rul, remaining_cycles)
                
                # 确保RUL不为负且有合理上限
                manual_rul = max(0, min(manual_rul, 200))
            
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
                    st.error("电池严重老化，建议尽快更换。")
            
            with col2:
                st.metric("剩余使用寿命 (RUL)", f"{manual_rul:.2f} 循环")
                
                # 添加RUL状态解释 - 混合版本
                if manual_rul > 50:
                    st.success("电池剩余寿命充足。")
                elif manual_rul > 20:
                    st.info("电池剩余寿命适中，可继续使用一段时间。")
                elif manual_rul > 0:
                    st.warning("电池剩余寿命较短，建议准备更换。")
                else:
                    st.error("电池已达到寿命终点，建议尽快更换。")
            
            # 创建并显示可视化
            manual_plot = create_prediction_plot(manual_soh, manual_rul)
            st.image(f"data:image/png;base64,{manual_plot}", use_column_width=True)
            
            # 显示计算详情
            st.markdown("### 计算详情")
            st.write(f"""
            - 初始容量: {initial_capacity:.2f} Ah
            - 当前容量: {current_capacity:.2f} Ah
            - 已完成循环: {cycles_completed} 循环
            - SOH: {manual_soh:.2f}%
            - {'SOH低于80%，RUL直接设为0' if manual_soh <= 80 else ''}
            """)
            
            if manual_soh > 80:
                st.write(f"""
                - 平均衰减率: {avg_decline_per_cycle:.4f}% / 循环
                - {'应用加速因子: ' + str(acceleration_factor):.2f if use_nonlinear_model_manual else '未应用加速因子'}
                - 预期未来衰减率: {future_decline_per_cycle:.4f}% / 循环
                - 预期总循环寿命: {expected_total_cycles_manual} 循环
                """)
    
    # 添加页脚
    st.markdown("---")
    st.markdown("© 2025 唐光盛-浙江锋锂团队& 基于机器学习的电池健康状态和剩余使用寿命预测")

if __name__ == "__main__":
    main()
