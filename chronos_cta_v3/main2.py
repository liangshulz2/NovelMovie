import pandas as pd
from data.futures_loader import load_futures
from model.chronos_model import ChronosModel
from model.xgb_model import XGBModel
from model.ensemble_model import bayesian_model_averaging
from alpha.alpha_engine import generate_signal
from backtest.backtest_engine import BacktestPlatform
from factors.factor_library import compute_factors
from factors.factor_selector import select_features

# ------------------- 参数 -------------------
symbol = "CU0"
start_date = "2025-08-01"
end_date = "2026-03-18"
MODEL_PATH = "E:/ai/chronos-2"
DEVICE = "cpu"

# ------------------- 1. 加载并截取日期段 -------------------
df = load_futures(symbol)
df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)

# ------------------- 2. 初始化模型 -------------------
chronos = ChronosModel(MODEL_PATH, DEVICE)
xgb = XGBModel()

# ------------------- 3. 生成每日 signal 序列 -------------------
signals = []
# 为了保证滚动窗口里有足够的历史数据，这里设定最小窗口为 21 天（可自行调节）
MIN_WINDOW = 21

for i in range(MIN_WINDOW, len(df)):
    # ----- 取历史窗口 -----
    hist = df.iloc[:i].copy()

    # ----- 计算因子（必须在特征选择前） -----
    hist = compute_factors(hist)

    # ----- 生成 target -----
    hist["target"] = hist["close"].pct_change().shift(-1)

    # ----- 去掉含 NaN 的行，确保特征选择有足够样本 -----
    clean_hist = hist.dropna(subset=["target"])
    if len(clean_hist) < 2:          # 至少需要 2 条记录才能训练
        continue

    # ----- 特征选择 -----
    feats = select_features(clean_hist, "target", top_n=30)

    # ----- 若特征选择返回空列表，使用默认基础特征作为回退 -----
    if not feats:
        feats = ["close", "open", "high", "low", "volume"]

    # ----- 构造训练 / 测试集 -----
    model_df = clean_hist[[*feats, "target"]].dropna()
    if len(model_df) < 2:            # 再次检查，防止因子缺失导致空 DataFrame
        continue

    x_train = model_df[feats].iloc[:-1]
    y_train = model_df["target"].iloc[:-1]
    x_test = model_df[feats].iloc[[-1]]

    # ----- 训练 XGB -----
    # XGBoost 需要二维数组，确保传入的对象不是空的
    xgb.fit(x_train.values, y_train.values)

    ml_pred = float(xgb.predict(x_test.values)[0])

    # ----- Chronos 预测 -----
    close = hist["close"].dropna()
    factor_mat = hist[feats].fillna(0)
    chronos_pred = chronos.predict(close, factor_mat, 1)
    chronos_alpha = (
        float(chronos_pred[0])
        if hasattr(chronos_pred, "__getitem__")
        else float(chronos_pred)
    )

    # ----- 贝叶斯加权得到 alpha -----
    target_last = model_df["target"].iloc[-1]   # 当天的真实 target
    alpha, model_weights = bayesian_model_averaging(
        predictions={"chronos": chronos_alpha, "xgb": ml_pred},
        errors={
            "chronos": abs(float(target_last - chronos_alpha)) + 1e-6,
            "xgb": abs(float(target_last - ml_pred)) + 1e-6,
        },
    )

    # ----- 生成交易信号（+1 / -1） -----
    signal = generate_signal(alpha)
    signals.append(signal)

# ------------------- 4. 对齐信号序列 -------------------
# 由于前面用了 MIN_WINDOW 天的滚动窗口，信号从第 MIN_WINDOW 天开始
df = df.iloc[MIN_WINDOW:].reset_index(drop=True)
df["signal"] = signals
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date')

# ------------------- 5. 回测 -------------------
bt = BacktestPlatform(train_window=120, test_window=20)

daily, trades = bt.run_with_trade_records(
    predictions=df["signal"],
    returns=df["close"].pct_change().fillna(0),
    prices=df["close"],
    #start_date=df["date"].iloc[0],  # 从信号开始的第一天
    #end_date=end_date,
)

print("回测每日结果：")
print(daily)
print("\n交易记录：")
print(trades)