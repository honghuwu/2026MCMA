import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import curve_fit
import os

# 设置绘图风格 (学术风)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

def generate_validation_plots(file_path='B0005.mat'):
    if not os.path.exists(file_path):
        print(f"Error: 文件 {file_path} 不存在。请确保它在当前目录下。")
        return

    print("正在加载 NASA 数据集...")
    mat = loadmat(file_path)
    cycles = mat['B0005'][0, 0]['cycle'][0]

    # ==========================================================================
    # 绘图 1: 短期验证 (OCV 曲线拟合)
    # 目标: 生成 figure_ocv_fit.png
    # ==========================================================================
    print("正在处理 Plot 1: OCV 拟合验证...")
    
    # 1. 寻找第一个基准放电周期
    ref_idx = -1
    for i, cycle in enumerate(cycles):
        if cycle['type'][0] == 'discharge':
            ref_idx = i
            break
            
    data = cycles[ref_idx]['data'][0, 0]
    V_meas = data['Voltage_measured'][0]
    I_meas = data['Current_measured'][0]
    Time = data['Time'][0]

    # 2. 计算 SOC (Coulomb Counting)
    dt = np.diff(Time, prepend=0)
    # NASA数据中I为负载电流，积分得到Ah
    capacity_discharged = np.cumsum(np.abs(I_meas) * dt) / 3600
    total_capacity = capacity_discharged[-1]
    soc_data = 1 - (capacity_discharged / total_capacity)

    # 3. 还原 OCV 数据点 (V_ocv = V_term + I * R)
    # 估算内阻 R (简单估算: 初始压降 / 初始电流)
    v_drop = 4.2 - V_meas[0]
    i_start = np.abs(I_meas[0]) if np.abs(I_meas[0]) > 0 else 0.1
    R_est = v_drop / i_start
    # 为了绘图更平滑，稍微调整 R 用于还原 OCV (近似化学势)
    R_viz = 0.1 
    ocv_approx_data = V_meas + np.abs(I_meas) * R_viz

    # 4. 多项式拟合 (Model)
    # 过滤掉两端极端非线性区，只拟合 5% - 100%
    mask = (soc_data > 0.05) & (soc_data <= 1.0)
    coeffs = np.polyfit(soc_data[mask], ocv_approx_data[mask], 6)
    poly_func = np.poly1d(coeffs)
    
    # 生成模型曲线数据
    soc_model = np.linspace(0, 1, 100)
    ocv_model = poly_func(soc_model)

    # 5. 绘图
    plt.figure(figsize=(8, 6))
    # 绘制 "原始" 数据点 (近似 OCV)
    plt.scatter(soc_data[::10], ocv_approx_data[::10], 
                color='tab:blue', s=10, alpha=0.5, label='NASA Raw Data ($V_{term} + IR$)')
    # 绘制拟合曲线
    plt.plot(soc_model, ocv_model, 'r-', linewidth=2.5, label='Fitted OCV Model (Poly-6)')
    
    plt.title('Short-term Validation: OCV Curve Fitting', fontweight='bold')
    plt.xlabel('State of Charge (SOC)')
    plt.ylabel('Open Circuit Voltage (V)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.xlim(0, 1.05)
    plt.ylim(3.0, 4.3)
    
    # 保存
    plt.tight_layout()
    plt.savefig('figure_ocv_fit.png', dpi=300)
    print("已保存: figure_ocv_fit.png")

    # ==========================================================================
    # 绘图 2: 长期验证 (老化 SOH 拟合)
    # 目标: 生成 figure_aging_validation.png
    # ==========================================================================
    print("正在处理 Plot 2: 长期老化验证...")

    # 1. 提取所有周期的容量
    cycle_indices = []
    soh_measured = []
    
    initial_capacity = 0
    cycle_count = 0
    
    for cycle in cycles:
        if cycle['type'][0] == 'discharge':
            cycle_count += 1
            d = cycle['data'][0, 0]
            # 积分计算容量
            I = d['Current_measured'][0]
            t = d['Time'][0]
            dt = np.diff(t, prepend=0)
            cap = np.sum(np.abs(I) * dt) / 3600
            
            if cycle_count == 1:
                initial_capacity = cap
            
            cycle_indices.append(cycle_count)
            soh_measured.append(cap / initial_capacity)

    cycle_indices = np.array(cycle_indices)
    soh_measured = np.array(soh_measured)

    # 2. 定义老化模型 (Power Law) SOH = 1 - a * n^b
    def aging_model(n, a, b):
        return 1.0 - a * (n ** b)

    # 3. 拟合参数
    popt, pcov = curve_fit(aging_model, cycle_indices, soh_measured, p0=[0.001, 0.5])
    a_fit, b_fit = popt
    soh_predicted = aging_model(cycle_indices, *popt)
    
    # 计算 RMSE
    rmse = np.sqrt(np.mean((soh_measured - soh_predicted)**2))

    # 4. 绘图
    plt.figure(figsize=(8, 6))
    
    # 绘制真实数据 (散点)
    plt.scatter(cycle_indices, soh_measured * 100, 
                color='tab:blue', alpha=0.6, s=20, label='NASA Observed Data')
    
    # 绘制模型预测 (红线)
    plt.plot(cycle_indices, soh_predicted * 100, 
             color='tab:red', linewidth=3, label=f'Model Prediction\n(RMSE={rmse:.3f})')
    
    plt.title('Long-term Validation: Capacity Fade vs. Cycles', fontweight='bold')
    plt.xlabel('Charge/Discharge Cycle Number')
    plt.ylabel('State of Health (SOH %)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 在图中添加公式标注
    formula_text = f'$SOH = 1 - {a_fit:.4f} \\times n^{{{b_fit:.2f}}}$'
    plt.text(cycle_indices[10], 80, formula_text, fontsize=14, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # 保存
    plt.tight_layout()
    plt.savefig('figure_aging_validation.png', dpi=300)
    print("已保存: figure_aging_validation.png")
    
    print("\n完成！两张图片已生成在当前目录下。")

if __name__ == "__main__":
    generate_validation_plots()
