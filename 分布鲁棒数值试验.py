import warnings
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def preprocess_data(price_file, wind_file, solar_file):
    print("--- 开始数据预处理 ---")

    try:
        df_price = pd.read_excel(price_file)
        df_wind = pd.read_excel(wind_file)
        df_solar = pd.read_excel(solar_file)
    except FileNotFoundError as e:
        print(f"错误：文件未找到 {e.filename}。请确保所有数据文件都在脚本所在目录下。")
        return None, None

    def create_scenario_matrix(df, data_col, time_col=None):
        if time_col is None:
            if '日期' not in df.columns or '时刻' not in df.columns:
                raise ValueError("电价数据文件中缺少 '日期' 或 '时刻' 列!")
            df['DateTime'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['时刻'].astype(str))
        else:
            if time_col not in df.columns:
                raise ValueError(f"风/光数据文件中缺少 '{time_col}' 列!")
            df.rename(columns={time_col: 'DateTime'}, inplace=True)
            df['DateTime'] = pd.to_datetime(df['DateTime'])

        df.rename(columns={data_col: 'Value'}, inplace=True)
        
        df.set_index('DateTime', inplace=True)
        hourly_data = df['Value'].resample('h').mean()
        
        hourly_df = hourly_data.to_frame()
        hourly_df['Date'] = hourly_df.index.date
        hourly_df['Hour'] = hourly_df.index.hour
        
        matrix = hourly_df.pivot(index='Hour', columns='Date', values='Value')
        matrix.dropna(axis=1, inplace=True)
        
        return matrix

    price_matrix = create_scenario_matrix(df_price, data_col='节点均价')
    wind_matrix = create_scenario_matrix(df_wind, data_col='Power (MW)', time_col='Time(year-month-day h:m:s)')
    solar_matrix = create_scenario_matrix(df_solar, data_col='Power (MW)', time_col='Time(year-month-day h:m:s)')
    
    num_scenarios = min(price_matrix.shape[1], wind_matrix.shape[1], solar_matrix.shape[1])
    
    if num_scenarios == 0:
        print("错误：处理后没有可用的完整天数场景。请检查原始数据是否包含至少一天的完整24小时数据。")
        return None, None
        
    print(f"数据处理完成。将使用 {num_scenarios} 个有效场景（天数）。")
    
    price_scenarios_df = price_matrix.iloc[:, :num_scenarios]
    wind_scenarios_df = wind_matrix.iloc[:, :num_scenarios]
    solar_scenarios_df = solar_matrix.iloc[:, :num_scenarios]
    
    renewable_scenarios_df = wind_scenarios_df + solar_scenarios_df
    
    return price_scenarios_df.values, renewable_scenarios_df.values

def get_model_parameters():
    params = {
        'T': 24,
        'C_G': 350,
        'C_curt': 50,
        'P_G_min': 0,
        'P_G_max': 50,
        'P_DA_max': 150,
        'beta': 0.3,
        'alpha': 0.95,
        'epsilon': 100
    }
    return params

def solve_deterministic_model(price_forecast, renewable_forecast, params):
    prob = LpProblem("Deterministic_VPP", LpMaximize)
    P_DA = LpVariable.dicts("P_DA", range(params['T']), 0, params['P_DA_max'])
    P_G = LpVariable.dicts("P_G", range(params['T']), params['P_G_min'], params['P_G_max'])
    p_w = LpVariable.dicts("p_w", range(params['T']), 0)
    p_curt = LpVariable.dicts("p_curt", range(params['T']), 0)
    prob += lpSum(price_forecast[t] * P_DA[t] - params['C_G'] * P_G[t] - params['C_curt'] * p_curt[t] for t in range(params['T']))
    for t in range(params['T']):
        prob += P_DA[t] == P_G[t] + p_w[t]
        prob += p_w[t] + p_curt[t] == renewable_forecast[t]
    prob.solve()
    num_vars = prob.numVariables()
    num_constrs = prob.numConstraints()
    if LpStatus[prob.status] == 'Optimal':
        return [P_DA[t].varValue for t in range(params['T'])], num_vars, num_constrs
    return None, num_vars, num_constrs

def solve_stochastic_model(price_scenarios, renewable_scenarios, params):
    S = price_scenarios.shape[1]
    prob = LpProblem("Stochastic_VPP", LpMaximize)
    P_DA = LpVariable.dicts("P_DA", range(params['T']), 0, params['P_DA_max'])
    P_G = LpVariable.dicts("P_G", range(params['T']), params['P_G_min'], params['P_G_max'])
    eta = LpVariable("eta")
    z = LpVariable.dicts("z", range(S), 0)
    p_w = LpVariable.dicts("p_w", [(t, s) for t in range(params['T']) for s in range(S)], 0)
    p_curt = LpVariable.dicts("p_curt", [(t, s) for t in range(params['T']) for s in range(S)], 0)
    profit_s = [lpSum(price_scenarios[t, s] * P_DA[t] - params['C_G'] * P_G[t] - params['C_curt'] * p_curt[(t, s)] for t in range(params['T'])) for s in range(S)]
    expected_profit = (1/S) * lpSum(profit_s)
    cvar = eta + (1 / (S * (1 - params['alpha']))) * lpSum(z)
    prob += (1 - params['beta']) * expected_profit - params['beta'] * cvar
    for s in range(S):
        prob += z[s] >= -profit_s[s] - eta
        for t in range(params['T']):
            prob += P_DA[t] == P_G[t] + p_w[(t, s)]
            prob += p_w[(t, s)] + p_curt[(t, s)] == renewable_scenarios[t, s]
    prob.solve()
    num_vars = prob.numVariables()
    num_constrs = prob.numConstraints()
    if LpStatus[prob.status] == 'Optimal':
        return [P_DA[t].varValue for t in range(params['T'])], num_vars, num_constrs
    return None, num_vars, num_constrs

def solve_dro_model(price_scenarios, renewable_scenarios, params):
    S = price_scenarios.shape[1]
    prob = LpProblem("DRO_VPP", LpMaximize)
    P_DA = LpVariable.dicts("P_DA", range(params['T']), 0, params['P_DA_max'])
    P_G = LpVariable.dicts("P_G", range(params['T']), params['P_G_min'], params['P_G_max'])
    eta = LpVariable("eta")
    z = LpVariable.dicts("z", range(S), 0)
    lambda_dro = LpVariable("lambda_dro", 0)
    y = LpVariable.dicts("y", range(S))
    p_w = LpVariable.dicts("p_w", [(t, s) for t in range(params['T']) for s in range(S)], 0)
    p_curt = LpVariable.dicts("p_curt", [(t, s) for t in range(params['T']) for s in range(S)], 0)
    profit_expr = [lpSum(price_scenarios[t, s] * P_DA[t] - params['C_G'] * P_G[t] - params['C_curt'] * p_curt[(t, s)] for t in range(params['T'])) for s in range(S)]
    prob += (1/S) * lpSum(y) - lambda_dro * params['epsilon']
    xi_vectors = np.concatenate([price_scenarios, renewable_scenarios], axis=0).T
    dist_matrix = np.linalg.norm(xi_vectors[:, np.newaxis, :] - xi_vectors[np.newaxis, :, :], ord=1, axis=2)
    for s in range(S):
        loss_s = -profit_expr[s]
        for j in range(S):
            prob += y[s] - y[j] <= lambda_dro * dist_matrix[s, j]
        prob += y[s] <= (1 - params['beta']) * profit_expr[s] - params['beta'] * (eta + z[s] / (1 - params['alpha']))
        prob += z[s] >= loss_s - eta
        for t in range(params['T']):
            prob += P_DA[t] == P_G[t] + p_w[(t, s)]
            prob += p_w[(t, s)] + p_curt[(t, s)] == renewable_scenarios[t, s]
    prob.solve()
    num_vars = prob.numVariables()
    num_constrs = prob.numConstraints()
    if LpStatus[prob.status] == 'Optimal':
        return [P_DA[t].varValue for t in range(params['T'])], num_vars, num_constrs
    return None, num_vars, num_constrs

def evaluate_strategy(p_da_strategy, price_scenarios, renewable_scenarios, params):
    S = price_scenarios.shape[1]
    realized_profits = []
    for s in range(S):
        profit_s = 0
        for t in range(params['T']):
            p_da_t = p_da_strategy[t]
            renewable_t_s = renewable_scenarios[t, s]
            price_t_s = price_scenarios[t, s]
            p_w_t_s = min(p_da_t, renewable_t_s)
            p_curt_t_s = renewable_t_s - p_w_t_s
            p_g_needed = p_da_t - p_w_t_s
            penalty = 0
            if p_g_needed > params['P_G_max']:
                penalty = 10 * params['C_G'] * (p_g_needed - params['P_G_max'])
                p_g_t_s = params['P_G_max']
            else:
                p_g_t_s = p_g_needed
            profit_s += price_t_s * p_da_t - params['C_G'] * p_g_t_s - params['C_curt'] * p_curt_t_s - penalty
        realized_profits.append(profit_s)
    return np.array(realized_profits)

if __name__ == "__main__":
    import matplotlib
    # 可选：强制使用MacOSX后端
    matplotlib.use('MacOSX')
    try:
        font_path = '/System/Library/Fonts/PingFang.ttc' 
        my_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = ['PingFang SC']
    except FileNotFoundError:
        print("警告：未找到指定中文字体，绘图可能出现乱码。请修改 'font_path' 变量。")
        my_font = None

    price_scenarios, renewable_scenarios = preprocess_data('广东节点价格数据.xlsx', 'Wind1.xlsx', 'Solar1.xlsx')
    if price_scenarios is None:
        exit()

    params = get_model_parameters()

    print("\n--- 开始求解确定性模型 ---")
    t0 = time.time()
    res_det = solve_deterministic_model(np.mean(price_scenarios, axis=1), np.mean(renewable_scenarios, axis=1), params)
    t_det = time.time() - t0
    if res_det[0] is not None:
        p_da_det, vars_det, cons_det = res_det
        print(f"确定性模型求解完成，耗时 {t_det:.2f}秒，变量数：{vars_det}，约束数：{cons_det}")
    else:
        print("确定性模型求解失败")
        p_da_det, vars_det, cons_det = None, 0, 0

    print("\n--- 开始求解随机规划模型 ---")
    t0 = time.time()
    res_sto = solve_stochastic_model(price_scenarios, renewable_scenarios, params)
    t_sto = time.time() - t0
    if res_sto[0] is not None:
        p_da_sto, vars_sto, cons_sto = res_sto
        print(f"随机规划模型求解完成，耗时 {t_sto:.2f}秒，变量数：{vars_sto}，约束数：{cons_sto}")
    else:
        print("随机规划模型求解失败")
        p_da_sto, vars_sto, cons_sto = None, 0, 0

    print("\n--- 开始求解分布鲁棒优化模型 ---")
    t0 = time.time()
    res_dro = solve_dro_model(price_scenarios, renewable_scenarios, params)
    t_dro = time.time() - t0
    if res_dro[0] is not None:
        p_da_dro, vars_dro, cons_dro = res_dro
        print(f"分布鲁棒优化模型求解完成，耗时 {t_dro:.2f}秒，变量数：{vars_dro}，约束数：{cons_dro}")
    else:
        print("分布鲁棒优化模型求解失败")
        p_da_dro, vars_dro, cons_dro = None, 0, 0

    # 打印模型规模与计算效率对比表
    print("\n" + "="*70)
    print("模型规模与计算效率对比")
    print("="*70)
    print(f"{'模型':<20}{'变量数':<12}{'约束数':<12}{'求解时间(秒)':<15}")
    print("-"*70)
    print(f"{'确定性模型':<20}{vars_det:<12}{cons_det:<12}{t_det:<15.2f}")
    print(f"{'随机规划模型':<20}{vars_sto:<12}{cons_sto:<12}{t_sto:<15.2f}")
    print(f"{'分布鲁棒模型':<20}{vars_dro:<12}{cons_dro:<12}{t_dro:<15.2f}")
    print("="*70)

    if p_da_det and p_da_sto and p_da_dro:
        profits_det = evaluate_strategy(p_da_det, price_scenarios, renewable_scenarios, params)
        profits_sto = evaluate_strategy(p_da_sto, price_scenarios, renewable_scenarios, params)
        profits_dro = evaluate_strategy(p_da_dro, price_scenarios, renewable_scenarios, params)

        results = {
            '确定性模型': {
                '期望利润': np.mean(profits_det),
                '利润标准差': np.std(profits_det),
                '最差情况利润': np.min(profits_det),
                'CVaR(alpha=0.95)': -np.mean(np.sort(profits_det)[:int(len(profits_det) * (1-params['alpha']))]) if len(profits_det) > 0 else 0
            },
            '随机规划模型': {
                '期望利润': np.mean(profits_sto),
                '利润标准差': np.std(profits_sto),
                '最差情况利润': np.min(profits_sto),
                'CVaR(alpha=0.95)': -np.mean(np.sort(profits_sto)[:int(len(profits_sto) * (1-params['alpha']))]) if len(profits_sto) > 0 else 0
            },
            '分布鲁棒模型': {
                '期望利润': np.mean(profits_dro),
                '利润标准差': np.std(profits_dro),
                '最差情况利润': np.min(profits_dro),
                'CVaR(alpha=0.95)': -np.mean(np.sort(profits_dro)[:int(len(profits_dro) * (1-params['alpha']))]) if len(profits_dro) > 0 else 0
            }
        }
        
        df_results = pd.DataFrame(results).T.round(2)
        df_results['求解用时(秒)'] = [t_det, t_sto, t_dro] 
        print("\n" + "="*20 + " 模型性能对比 " + "="*20)
        print(df_results)
        print("="*54)

        plt.figure(figsize=(12, 6))
        plt.plot(p_da_det, marker='o', linestyle='-', label='确定性模型')
        plt.plot(p_da_sto, marker='s', linestyle='--', label='随机规划模型')
        plt.plot(p_da_dro, marker='^', linestyle='-.', label='分布鲁棒模型')
        plt.title('不同模型下的日前申报出力曲线', fontproperties=my_font, fontsize=16)
        plt.xlabel('时段 (小时)', fontproperties=my_font, fontsize=12)
        plt.ylabel('申报出力 (MW)', fontproperties=my_font, fontsize=12)
        plt.xticks(range(params['T']))
        plt.grid(True)
        plt.legend(prop=my_font)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.boxplot([profits_det, profits_sto, profits_dro], labels=['确定性', '随机规划', '分布鲁棒'])
        plt.title('不同策略下的利润分布', fontproperties=my_font, fontsize=16)
        plt.ylabel('日利润 (元)', fontproperties=my_font, fontsize=12)
        ax = plt.gca()
        ax.set_xticklabels(['确定性', '随机规划', '分布鲁棒'], fontproperties=my_font)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
    else:
        print("\n错误：一个或多个模型求解失败，无法进行结果对比。")
