"""
2026 MCM Problem A - Smartphone Battery Modeling
Team Code: Antigravity-User-V1
Model Type: Data-Driven Continuous-Time Thevenin Model
Data Source: NASA Prognostics Data Repository (Battery B0005)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.io import loadmat
import os

# ==============================================================================
# 第一部分：NASA 数据解析器 (Data Parser)
# 功能：从原始 MAT 文件中提取物理参数 (OCV曲线 和 内阻)
# ==============================================================================
class NasaDataLoader:
    def __init__(self, file_path='B0005.mat'):
        self.file_path = file_path
        self.ocv_func = None 
        self.R_est = 0.08  # 给一个合理的默认值 (0.08 Ohm)
        self._load_and_process()

    def _load_and_process(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"找不到 {self.file_path}")
            
        mat = loadmat(self.file_path)
        cycles = mat['B0005'][0, 0]['cycle'][0]
        
        # 寻找第1个放电周期
        ref_idx = -1
        for i, cycle in enumerate(cycles):
            if cycle['type'][0] == 'discharge':
                ref_idx = i
                break
        
        data = cycles[ref_idx]['data'][0, 0]
        V_meas = data['Voltage_measured'][0]
        I_meas = data['Current_measured'][0]
        Time = data['Time'][0]
        
        # --- 修复点 1: 智能计算内阻 ---
        # 不再盲目取第0个点，而是寻找电流剧烈变化的时刻
        # NASA数据中，放电电流通常是从 0 跳变到 -2A (或其他值)
        # 我们寻找电流幅值超过 0.5A 的第一个点
        load_idx = np.where(np.abs(I_meas) > 0.5)[0]
        
        if len(load_idx) > 0:
            idx = load_idx[0] 
            # 比较负载前一刻(静置)和负载时刻的电压差
            # 如果 idx=0，说明数据一开始就有负载，那就假设开路是 4.2V
            v_open = V_meas[idx-1] if idx > 0 else 4.2
            v_load = V_meas[idx]
            i_load = abs(I_meas[idx])
            
            R_calc = abs(v_open - v_load) / i_load
            # 增加一个安全范围限制 (0.01 - 0.2 Ohm)，防止异常数据
            self.R_est = max(0.01, min(R_calc, 0.2))
        
        print(f"[*] 修正后的内阻 R0: {self.R_est:.4f} Ohms")
        
        # --- 计算 OCV 曲线 (保持不变) ---
        dt = np.diff(Time, prepend=0)
        discharged_ah = np.cumsum(I_meas * dt) / 3600 
        soc_array = 1 - (discharged_ah / discharged_ah[-1])
        ocv_approx = V_meas + I_meas * self.R_est
        mask = (soc_array > 0.05) & (soc_array <= 1.0)
        p_coeffs = np.polyfit(soc_array[mask], ocv_approx[mask], 6)
        self.ocv_func = np.poly1d(p_coeffs)
# ==============================================================================
# 第二部分：连续时间物理模型 (Physics-Based Model)
# 功能：定义微分方程组 (ODEs)
# ==============================================================================
class SmartphoneModel:
    def __init__(self, nasa_loader, scaling_factor=2.0):
        """
        scaling_factor: NASA电池约 2Ah，现代手机约 4Ah，因此放大 2.0 倍
        """
        self.loader = nasa_loader
        self.Q_total = 2.0 * 3600 * scaling_factor # 总容量 (Coulombs)
        
        # 物理参数
        self.mass = 0.05   # kg (电池质量)
        self.Cp = 900.0    # J/kgK (比热容)
        self.h_conv = 5.0  # W/m2K (散热系数，手机被动散热较差)
        self.Area = 0.012  # m2 (表面积)
        
        # 电路参数 (Thevenin 模型)
        self.R1 = 0.02     # 极化电阻
        self.C1 = 1000.0   # 极化电容 (大电容模拟电压恢复慢)
        self.V_cutoff = 3.2 # 关机电压

    def get_R0(self, T):
        """内阻随温度变化的 Arrhenius 模型"""
        T_ref = 298.15 # 25 C
        Ea_R = 1500.0  # 活化能/气体常数 (经验值)
        # T 越低，R 越大
        return self.loader.R_est * np.exp(Ea_R * (1/T - 1/T_ref))

    def derivatives(self, t, y, load_func):
        """
        核心微分方程组
        y[0]: SOC (0-1)
        y[1]: Vp (极化电压)
        y[2]: T (电池温度 Kelvin)
        """
        SOC, Vp, T = y
        P_demand = load_func(t) # 获取当前时刻用户功率
        
        # 1. 获取物理参数
        R0 = self.get_R0(T)
        OCV = self.loader.ocv_func(np.clip(SOC, 0, 1))
        
        # 2. 计算电流 I (代数环求解)
        # 方程: P = V_term * I = (OCV - Vp - I*R0) * I
        # 整理得: R0*I^2 - (OCV-Vp)*I + P = 0
        V_eff = OCV - Vp
        delta = V_eff**2 - 4 * R0 * P_demand
        
        if delta < 0:
            # 电压崩溃 (所需功率超过电池极限)
            I = V_eff / (2*R0)
        else:
            # 取较小电流根
            I = (V_eff - np.sqrt(delta)) / (2 * R0)
            
        # 3. 状态方程
        # dSOC/dt = -I / Q
        dSOC = -I / self.Q_total
        
        # dVp/dt = -Vp/R1C1 + I/C1 (极化动态)
        dVp = -Vp / (self.R1 * self.C1) + I / self.C1
        
        # dT/dt = (Heat_Gen - Heat_Loss) / mCp
        Q_gen = I**2 * (R0 + self.R1)
        Q_loss = self.h_conv * self.Area * (T - 298.15)
        dT = (Q_gen - Q_loss) / (self.mass * self.Cp)
        
        return [dSOC, dVp, dT]

# ==============================================================================
# 第三部分：仿真控制与用户行为 (Simulation)
# ==============================================================================

# 定义用户负载曲线 (User Profile)
def user_activity(t):
    """
    模拟一天的手机使用 (Watts):
    - 0-8h: 睡眠 (0.1W)
    - 8-9h: 通勤/导航 (3.0W)
    - 9-12h: 待机/偶尔消息 (0.5W)
    - 12-13h: 大型游戏 (5.0W) !!! 高负载
    - 13-18h: 工作 (0.5W)
    - 18-20h: 视频流 (3.0W)
    """
    h = (t / 3600) % 24
    noise = np.random.normal(0, 0.2) # 添加随机噪声
    return (np.sin( h * np.pi / 6)  + 1.0 + noise)
    if 0 <= h < 8: return 0.1
    if 8 <= h < 9: return 3.0 + noise
    if 9 <= h < 12: return 0.5 + noise
    if 12 <= h < 13: return 5.0 + noise # 高功耗时刻
    if 13 <= h < 18: return 0.5 + noise
    if 18 <= h < 20: return 3.0 + noise
    return 0.8 + noise # 晚上刷手机

def main():
    # 1. 加载数据
    try:
        loader = NasaDataLoader('B0005.mat')
    except Exception as e:
        print(e)
        return

    # 2. 初始化模型
    model = SmartphoneModel(loader, scaling_factor=2.2) # 模拟 4400mAh 电池
    
    # 3. 设定求解器
    t_span = (0, 24*3600) # 模拟 24 小时
    y0 = [1.0, 0.0, 298.15] # 初始: 100%电量, 0V极化, 25度
    
    # 定义截止事件 (TTE): 当电压低于 3.2V 时停止
    def cutoff_event(t, y):
        SOC, Vp, T = y
        R0 = model.get_R0(T)
        P = user_activity(t)
        OCV = model.loader.ocv_func(np.clip(SOC, 0, 1))
        
        V_eff = OCV - Vp
        delta = V_eff**2 - 4*R0*P
        if delta < 0: return -1 # 崩溃
        I = (V_eff - np.sqrt(delta))/(2*R0)
        V_term = V_eff - I*R0
        return V_term - model.V_cutoff
    
    cutoff_event.terminal = True # 触发即停止
    cutoff_event.direction = -1

    print("[*] 开始求解微分方程...")
    sol = solve_ivp(
        fun=lambda t, y: model.derivatives(t, y, user_activity),
        t_span=t_span,
        y0=y0,
        method='LSODA', # 刚性方程推荐算法
        events=cutoff_event,
        max_step=60 # 限制步长以捕捉瞬间变化
    )
    
    # 4. 结果可视化
    print(f"[*] 仿真结束。手机坚持了: {sol.t[-1]/3600:.2f} 小时")
    plot_results(sol, model)

def plot_results(sol, model):
    t_h = sol.t / 3600
    soc = sol.y[0]
    temp = sol.y[2] - 273.15
    
    # 反算电压用于绘图
    voltages = []
    powers = []
    for i, t in enumerate(sol.t):
        P = user_activity(t)
        powers.append(P)
        # 计算电压 (简化版)
        R0 = model.get_R0(sol.y[2][i])
        OCV = model.loader.ocv_func(np.clip(soc[i], 0, 1))
        V_eff = OCV - sol.y[1][i]
        delta = V_eff**2 - 4*R0*P
        if delta >= 0:
            I = (V_eff - np.sqrt(delta))/(2*R0)
            voltages.append(V_eff - I*R0)
        else:
            voltages.append(0)

    # 绘图设置
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # 图1: 用户负载
    axs[0].fill_between(t_h, powers, color='gray', alpha=0.3)
    axs[0].plot(t_h, powers, 'k', linewidth=1)
    axs[0].set_ylabel('Power (W)')
    axs[0].set_title('Figure 1: User Activity Profile (Power Demand)')
    axs[0].grid(True)
    
    # 图2: SOC
    axs[1].plot(t_h, soc*100, 'b', linewidth=2)
    axs[1].set_ylabel('SOC (%)')
    axs[1].set_title('Figure 2: State of Charge (Remaining Capacity)')
    axs[1].grid(True)
    
    # 图3: 端电压 (关键验证)
    axs[3].plot(t_h, voltages, 'g', linewidth=2)
    axs[3].axhline(model.V_cutoff, color='r', linestyle='--', label='Cutoff 3.2V')
    axs[3].set_ylabel('Voltage (V)')
    axs[3].set_title('Figure 3: Terminal Voltage (Shows Voltage Sag)')
    axs[3].legend()
    axs[3].grid(True)
    
    # 图4: 温度
    axs[2].plot(t_h, temp, 'r', linewidth=2)
    axs[2].set_ylabel('Temp (°C)')
    axs[2].set_title('Figure 4: Battery Temperature')
    axs[2].grid(True)
    
    plt.xlabel('Time (Hours)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
