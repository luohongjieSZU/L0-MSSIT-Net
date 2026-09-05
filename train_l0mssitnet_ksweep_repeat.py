

import argparse
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta as rd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from l0_MSSIT_Net import l0_MSSIT_Net
from SSPO import SSPO_fun
from Min_Var import min_var
from Mean_Var import mean_var


df = pd.read_csv('data.csv').set_index('Date')
df_sp = pd.read_csv('sp500.csv').set_index('Date')

N_STOCKS = df.shape[1]          # 471
DEVICE = torch.device('cpu')
RBP = 1
NUM_EPOCHS = 1000
LR = 1e-4
COST_RATE = 0.001
TRAIN_DAYS = 120
TEST_DAYS = 19

K_VALUES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
SEEDS = [1024, 2048, 42, 123, 456, 789, 2023, 2024, 999, 8888]
M_DAY_LIST = [5, 10, 15, 20, 25, 30]


def date_slicer(df, start, duration, rebalancing_period=0):
    start_ = str(datetime.strptime(start, '%Y-%m-%d').date() + rd(months=rebalancing_period))
    end_ = str(datetime.strptime(start_, '%Y-%m-%d').date() + rd(months=duration) - rd(days=1))
    return df.loc[start_:end_]


def data_process(df):

    df = df.pct_change().tail(-1)
    df = df + 1
    df = df.cumprod()
    df = df - 1
    return torch.from_numpy(df.iloc[-1, :].to_numpy()).type(torch.Tensor)


def data_process_m(df, m):

    df = df.pct_change().tail(-1).tail(m)
    df = df + 1
    df = df.cumprod()
    df = df - 1
    return torch.from_numpy(df.to_numpy()).type(torch.Tensor)


def daily_change(df):

    return df.pct_change().tail(-1)


def mday_cw(df, m):

    df = df.pct_change().tail(-m)
    df = df + 1
    df = df.cumprod()
    df = df - 1
    return torch.from_numpy(df.iloc[-1, :].to_numpy()).type(torch.Tensor)


def weight_mday_cw(df, w, m):

    df = df.pct_change()
    df = df.tail(m)
    df = df.to_numpy()
    weighted_returns = torch.tensor(df, dtype=torch.float32) * w
    cumulative_returns = torch.cumprod(1 + weighted_returns, dim=0) - 1
    return cumulative_returns[-1].clone().detach().float()


def calculate_portfolio_returns(x, weights):
    return np.dot(x.values, weights)


def _y_scalar(y_row):

    if isinstance(y_row, pd.Series):
        return float(y_row.iloc[0])
    return float(y_row)


def RMSE(x, y, weights):
    temp = 0
    for i in range(len(x)):
        weighted_sum = np.dot(x.iloc[i].values, weights)
        temp += (weighted_sum - _y_scalar(y.iloc[i])) ** 2
    return math.sqrt(temp / len(x))


def annualized_return(x, weights, trading_days=252):
    port_ret = calculate_portfolio_returns(x, weights)
    n_days = len(port_ret)
    if n_days == 0:
        return 0.0
    total_growth = np.prod(1 + port_ret)
    return float((total_growth ** (trading_days / n_days)) - 1)


def sharpe_ratio(x, weights, risk_free_rate=0.02, trading_days=252):
    port_ret = calculate_portfolio_returns(x, weights)
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess_returns = port_ret - daily_rf
    mean_ex = np.mean(excess_returns)
    std_ret = np.std(port_ret, ddof=1)
    if std_return_guard(std_ret):
        return 0.0
    return float((mean_ex / std_ret) * math.sqrt(trading_days))


def max_drawdown(x, weights):
    port_ret = calculate_portfolio_returns(x, weights)
    cum_returns = np.cumprod(1 + port_ret)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / running_max
    return float(np.max(drawdowns))


def std_return_guard(std_ret):

    return std_ret == 0 or np.isnan(std_ret)


def tracking_mse(weights, x_cum, y_cum):
    x_daily = (x_cum[1:] + 1) / (x_cum[:-1] + 1) - 1
    y_daily = (y_cum[1:] + 1) / (y_cum[:-1] + 1) - 1
    y_daily = y_daily.reshape(-1)
    port_daily = x_daily.matmul(weights)
    return torch.nn.functional.mse_loss(port_daily, y_daily)


def compute_transaction_cost(current_weights, previous_weights, cost_rate=0.001):


    def to_1d_np(w):
        if torch.is_tensor(w):
            return w.detach().cpu().numpy().squeeze().flatten()
        return np.asarray(w).squeeze().flatten()

    if previous_weights is None:
        return 0.0, 0.0

    delta = to_1d_np(current_weights) - to_1d_np(previous_weights)
    turnover = float(0.5 * np.sum(np.abs(delta)))
    transaction_cost = cost_rate * turnover
    return transaction_cost, turnover


def RMSE_with_trade(x, y, weights, transaction_cost=0.0):
    temp = 0
    for i in range(len(x)):
        weighted_sum = np.dot(x.iloc[i].values, weights)
        if i == 0:
            weighted_sum -= transaction_cost
        temp += (weighted_sum - _y_scalar(y.iloc[i])) ** 2
    return math.sqrt(temp / len(x))


def calculate_portfolio_returns_with_trade(x, weights, transaction_cost=0.0):
    port_ret = np.dot(x.values, weights)
    port_ret[0] = port_ret[0] - transaction_cost
    return port_ret


def annualized_return_with_trade(x, weights, trading_days=252, transaction_cost=0.0):
    port_ret = calculate_portfolio_returns_with_trade(x, weights, transaction_cost)
    n_days = len(port_ret)
    if n_days == 0:
        return 0.0
    total_growth = np.prod(1 + port_ret)
    return float((total_growth ** (trading_days / n_days)) - 1)


def sharpe_ratio_with_trade(x, weights, risk_free_rate=0.02, trading_days=252, transaction_cost=0.0):
    port_ret = calculate_portfolio_returns_with_trade(x, weights, transaction_cost)
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess_returns = port_ret - daily_rf
    mean_ex = np.mean(excess_returns)
    std_ret = np.std(port_ret, ddof=1)
    if std_return_guard(std_ret):
        return 0.0
    return float((mean_ex / std_ret) * math.sqrt(trading_days))


def max_drawdown_with_trade(x, weights, transaction_cost=0.0):
    port_ret = calculate_portfolio_returns_with_trade(x, weights, transaction_cost)
    cum_returns = np.cumprod(1 + port_ret)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / running_max
    return float(np.max(drawdowns))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_window_cache(n_windows):

    cache = []
    for i in range(n_windows):
        print(f"  预生成窗口 {i + 1}/{n_windows} 的数据与特征...", flush=True)
        # ----- 数据切片（与 train_SP500_lhj.py 2.1 节一致） -----
        x_train = data_process(date_slicer(df, '2014-08-01', 36, i * RBP))
        x_train_new = data_process_m(date_slicer(df, '2014-08-01', 36, i * RBP), TRAIN_DAYS)
        y_train_new = data_process_m(date_slicer(df_sp, '2014-08-01', 36, i * RBP), TRAIN_DAYS)
        y_train_new = y_train_new.squeeze(-1) if y_train_new.dim() > 1 else y_train_new
        x_test_m = data_process_m(date_slicer(df, '2016-07-01', 1, i * RBP), TEST_DAYS)
        # 评估用 6 个月日收益
        x_change = daily_change(date_slicer(df, '2016-07-01', 6, i * RBP))
        y_change = daily_change(date_slicer(df_sp, '2016-07-01', 6, i * RBP))

        # ----- 基础组合权重（特征来源, 与 train_SP500_lhj.py 2.2 节一致） -----
        sspo_weight = SSPO_fun(x_train)
        minvar_weight = min_var(x_train_new)
        meanvar_weigth = mean_var(x_train_new)
        equal_weight = torch.ones_like(x_train) / N_STOCKS

        # ----- 30 维特征矩阵 q_t: cw(6) + sspo(6) + minvar(6) + equal(6) + meanvar(6) -----
        df_slice = date_slicer(df, '2014-08-01', 36, i * RBP)
        feats = [mday_cw(df_slice, m) for m in M_DAY_LIST]
        for w in [sspo_weight, minvar_weight, equal_weight, meanvar_weigth]:
            feats.extend(weight_mday_cw(df_slice, w, m) for m in M_DAY_LIST)
        q_t = torch.stack(feats)   # [30, 471]

        cache.append({
            'x_train_new': x_train_new, 'y_train_new': y_train_new,
            'x_test_m': x_test_m, 'q_t': q_t,
            'x_change': x_change, 'y_change': y_change,
        })
    return cache

def train_one_window(model, optimizer, wc):
    model.reset_parameters()
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()

    final_loss = None
    for epoch in range(NUM_EPOCHS):
        weights = model(wc['x_train_new'], wc['q_t'])
        loss = tracking_mse(weights, wc['x_train_new'], wc['y_train_new'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    return final_loss


def test_one_window(model, wc, previous_weights):
    t0 = time.time()
    with torch.no_grad():
        weights = model(wc['x_test_m'], wc['q_t']).cpu().detach().numpy().copy()
    testing_time = time.time() - t0

    cost, turnover = compute_transaction_cost(weights, previous_weights, COST_RATE)

    rmse = RMSE(wc['x_change'], wc['y_change'], weights)
    ann_ret = annualized_return(wc['x_change'], weights)
    sharpe = sharpe_ratio(wc['x_change'], weights)
    mdd = max_drawdown(wc['x_change'], weights)

    rmse_t = RMSE_with_trade(wc['x_change'], wc['y_change'], weights, cost)
    ann_t = annualized_return_with_trade(wc['x_change'], weights, transaction_cost=cost)
    sharpe_t = sharpe_ratio_with_trade(wc['x_change'], weights, transaction_cost=cost)
    mdd_t = max_drawdown_with_trade(wc['x_change'], weights, cost)

    test_dic = {
        'RMSE': rmse, 'annualized_return': ann_ret,
        'sharpe_ratio': sharpe, 'Max Drawdown': mdd, 'turnover': turnover,
        'RMSE_trade': rmse_t, 'annualized_return_trade': ann_t,
        'sharpe_ratio_trade': sharpe_t, 'Max Drawdown_trade': mdd_t,
    }
    return test_dic, testing_time, weights


def summarize_results(results_list, times_list):
    df_ = pd.DataFrame(results_list)

    def safe_col(col, func):
        if col not in df_.columns:
            print(f"[summarize_results 警告] 缺少 '{col}', 记为 NaN")
            return float('nan')
        return float(func(df_[col]))

    return {
        "best_rmse": safe_col("RMSE", lambda s: s.min()),
        "average_rmse": safe_col("RMSE", lambda s: s.mean()),
        "Annualized Return": safe_col("annualized_return", lambda s: s.mean()),
        "Sharpe Ratio": safe_col("sharpe_ratio", lambda s: s.mean()),
        "Max Drawdown": safe_col("Max Drawdown", lambda s: s.mean()),
        "best_rmse_trade": safe_col("RMSE_trade", lambda s: s.min()),
        "average_rmse_trade": safe_col("RMSE_trade", lambda s: s.mean()),
        "Annualized Return_trade": safe_col("annualized_return_trade", lambda s: s.mean()),
        "Sharpe Ratio_trade": safe_col("sharpe_ratio_trade", lambda s: s.mean()),
        "Max Drawdown_trade": safe_col("Max Drawdown_trade", lambda s: s.mean()),
        "turnover": safe_col("turnover", lambda s: s.mean()),
        "test_time": float(np.mean(times_list)) if times_list else 0.0,
    }


LOG_COLUMNS = [
    'dataset', 'model', 'time', 'LR', 'epoches', 'rbp', 'sparsity', 'seed',
    'best_rmse', 'average_rmse',
    'Annualized Return', 'Sharpe Ratio', 'Max Drawdown',
    'best_rmse_trade', 'average_rmse_trade',
    'Annualized Return_trade', 'Sharpe Ratio_trade', 'Max Drawdown_trade',
    'turnover', 'test_time',
]

WINDOW_COLUMNS = [
    'k', 'seed', 'window',
    'RMSE', 'annualized_return', 'sharpe_ratio', 'Max Drawdown', 'turnover',
    'RMSE_trade', 'annualized_return_trade', 'sharpe_ratio_trade', 'Max Drawdown_trade',
    'test_time',
]

SUMMARY_COLUMNS = [
    'k', 'n_seeds',
    'best_rmse_mean', 'best_rmse_std',
    'average_rmse_mean', 'average_rmse_std',
    'Annualized Return_mean', 'Sharpe Ratio_mean', 'Max Drawdown_mean',
    'turnover_mean', 'test_time_mean',
    'best_rmse_trade_mean', 'average_rmse_trade_mean', 'Sharpe Ratio_trade_mean',
]


def ensure_csv(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    existing = pd.read_csv(path)
    if list(existing.columns) != columns:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        backup = path.replace('.csv', f'_pre_{stamp}.csv')
        os.rename(path, backup)
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def run_one_config(k, seed, window_cache, log_path, windows_path):
    setup_seed(seed)
    model = l0_MSSIT_Net(k=k).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\n{'=' * 70}")
    print(f"k={k} | Seed={seed} | Windows={len(window_cache)}")
    print(f"{'=' * 70}")

    prev_weights = None
    all_results, all_times, all_window_rows = [], [], []

    for i, wc in enumerate(window_cache):
        t_win = time.time()
        train_one_window(model, optimizer, wc)
        dic, t_test, weights = test_one_window(model, wc, prev_weights)
        prev_weights = weights
        all_results.append(dic)
        all_times.append(t_test)

        n_nonzero = int(np.sum(weights > 0))
        all_window_rows.append({
            'k': k, 'seed': seed, 'window': i + 1,
            'RMSE': dic['RMSE'], 'annualized_return': dic['annualized_return'],
            'sharpe_ratio': dic['sharpe_ratio'], 'Max Drawdown': dic['Max Drawdown'],
            'turnover': dic['turnover'],
            'RMSE_trade': dic['RMSE_trade'],
            'annualized_return_trade': dic['annualized_return_trade'],
            'sharpe_ratio_trade': dic['sharpe_ratio_trade'],
            'Max Drawdown_trade': dic['Max Drawdown_trade'],
            'test_time': t_test,
        })
        print(f"  W{i + 1:02d}/{len(window_cache)} | RMSE={dic['RMSE']:.6f} "
              f"| RMSE_trade={dic['RMSE_trade']:.6f} | 非零={n_nonzero}/{k} "
              f"| 换手={dic['turnover']:.4f} | 本窗耗时={time.time() - t_win:.1f}s", flush=True)

    windows_df = pd.DataFrame(all_window_rows, columns=WINDOW_COLUMNS)
    windows_df.to_csv(windows_path, mode='a', header=not os.path.exists(windows_path), index=False)

    summary = summarize_results(all_results, all_times)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_row = {
        'dataset': 'SP500', 'model': 'l0_MSSIT_Net', 'time': timestamp,
        'LR': LR, 'epoches': NUM_EPOCHS, 'rbp': RBP, 'sparsity': k, 'seed': seed,
        **summary,
    }
    log_df = pd.DataFrame([log_row], columns=LOG_COLUMNS)
    log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

    print(f"\n{'─' * 50}")
    print(f"[k={k} | Seed {seed}] 汇总:")
    print(f"  best RMSE:    {summary['best_rmse']:.6f} | trade: {summary['best_rmse_trade']:.6f}")
    print(f"  avg RMSE:     {summary['average_rmse']:.6f} | trade: {summary['average_rmse_trade']:.6f}")
    print(f"  Sharpe:       {summary['Sharpe Ratio']:.4f} | trade: {summary['Sharpe Ratio_trade']:.4f}")
    print(f"  Ann.Ret:      {summary['Annualized Return']:.4f} | Max DD: {summary['Max Drawdown']:.4f}")
    print(f"  Turnover:     {summary['turnover']:.4f} | test time: {summary['test_time']:.4f}s")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='l0-MSSIT-Net 稀疏度扫描重复实验')
    parser.add_argument('--quick', action='store_true', help='快速自检: k∈{10,50} × 1 种子 × 2 窗口')
    parser.add_argument('--ks', type=int, nargs='+', default=None,
                        help='只跑指定稀疏度, 如 --ks 30 50 (默认全部 9 个)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='只跑指定种子 (默认全部 10 个)')
    parser.add_argument('--windows', type=int, default=None,
                        help='窗口数量覆盖 (默认 24)')
    args = parser.parse_args()

    ks_to_run = [10, 50] if args.quick else (args.ks if args.ks else K_VALUES)
    seeds_to_run = [SEEDS[0]] if args.quick else (args.seeds if args.seeds else SEEDS)
    n_windows = 2 if args.quick else (args.windows if args.windows else int(24 / RBP))

    result_dir = os.path.join(BASE_DIR, 'result')
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, 'l0mssitnet_ksweep_logs.csv')
    windows_path = os.path.join(result_dir, 'l0mssitnet_ksweep_windows.csv')
    summary_path = os.path.join(result_dir, 'l0mssitnet_ksweep_summary.csv')

    ensure_csv(log_path, LOG_COLUMNS)
    ensure_csv(windows_path, WINDOW_COLUMNS)


    print(f"预生成 {n_windows} 个窗口的数据与特征矩阵 q_t ...")
    t_cache = time.time()
    window_cache = build_window_cache(n_windows)
    print(f"窗口数据缓存完成, 耗时 {time.time() - t_cache:.1f} 秒")


    t_start = time.time()
    per_config_summaries = []   # [(k, seed, summary), ...]
    for k in ks_to_run:
        for seed in seeds_to_run:
            summary = run_one_config(k, seed, window_cache, log_path, windows_path)
            per_config_summaries.append((k, seed, summary))


    summary_rows = []
    for k in ks_to_run:
        ks_summaries = [s for (kk, _, s) in per_config_summaries if kk == k]
        if not ks_summaries:
            continue
        def agg(col, fn):
            vals = [s[col] for s in ks_summaries if not math.isnan(s[col])]
            return float(fn(vals)) if vals else float('nan')
        summary_rows.append({
            'k': k, 'n_seeds': len(ks_summaries),
            'best_rmse_mean': agg('best_rmse', np.mean), 'best_rmse_std': agg('best_rmse', np.std),
            'average_rmse_mean': agg('average_rmse', np.mean), 'average_rmse_std': agg('average_rmse', np.std),
            'Annualized Return_mean': agg('Annualized Return', np.mean),
            'Sharpe Ratio_mean': agg('Sharpe Ratio', np.mean),
            'Max Drawdown_mean': agg('Max Drawdown', np.mean),
            'turnover_mean': agg('turnover', np.mean),
            'test_time_mean': agg('test_time', np.mean),
            'best_rmse_trade_mean': agg('best_rmse_trade', np.mean),
            'average_rmse_trade_mean': agg('average_rmse_trade', np.mean),
            'Sharpe Ratio_trade_mean': agg('Sharpe Ratio_trade', np.mean),
        })
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"实验完成! 总耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    print(f"汇总日志:   {log_path}")
    print(f"窗口明细:   {windows_path}")
    print(f"跨种子聚合: {summary_path}")
    if summary_rows:
        print(f"\n{'k':>4} | {'avgRMSE':>9} | {'bestRMSE':>9} | {'Sharpe':>7} | {'Turnover':>8} | {'avgRMSE_tc':>9}")
        for r in summary_rows:
            print(f"{r['k']:>4} | {r['average_rmse_mean']:>9.6f} | {r['best_rmse_mean']:>9.6f} "
                  f"| {r['Sharpe Ratio_mean']:>7.4f} | {r['turnover_mean']:>8.4f} "
                  f"| {r['average_rmse_trade_mean']:>9.6f}")
