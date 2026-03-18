# NovelMovie / Chronos CTA V3 使用说明（中文）

本文档给出该项目的完整上手流程，包括：
- 环境准备
- 配置项说明
- 主程序运行方法
- `backtest_engine` 回测示例（基础回测 + 滚动窗口回测 + 交易记录）
- 常见问题排查

---

## 1. 项目简介

这是一个面向期货场景的 CTA 策略研究框架，主要流程为：
1. 通过 AkShare 拉取期货行情；
2. 计算因子并做特征筛选；
3. 使用 Chronos + XGBoost 进行预测，并做模型融合；
4. 通过风险与仓位模块给出最终仓位；
5. 使用 `backtest/backtest_engine.py` 进行策略回测与交易记录生成。

目录结构（核心）：

```text
chronos_cta_v3/
  main.py                    # 主流程入口
  config.py                  # 全局配置
  data/futures_loader.py     # 期货数据加载
  factors/                   # 因子计算与筛选
  model/                     # Chronos/XGB 模型与融合
  portfolio/                 # 仓位、风控、组合优化
  risk/                      # 风险计算
  backtest/backtest_engine.py# 回测引擎
```

---

## 2. 环境准备

建议 Python 3.10+。

### 2.1 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

> 说明：
> - `akshare` 用于行情下载；
> - `torch` 与 `chronos-forecasting` 用于 Chronos 模型；
> - `xgboost` 用于机器学习预测。

如你的运行环境有 CUDA，可根据自身环境安装对应版本的 `torch`。

### 2.2 安装后快速验证（建议首次必做）

```bash
python -V
pip -V
pip install -r requirements.txt
pytest -q
```

若失败，请按下面顺序排查：
- `python -V` 与 `pip -V` 显示的 Python 主版本是否一致（避免装到错误环境）；
- 先升级打包工具：`python -m pip install -U pip setuptools wheel`；
- 若 `torch`/`chronos-forecasting` 安装失败，先确认本机是否需要 CUDA 版本并按官方指引安装；
- 若 `pytest -q` 失败，优先看首个报错模块并确认依赖是否完整安装。

### 2.3 模型准备

建议通过 `.env` 或系统环境变量配置（示例）：

```bash
export CHRONOS_MODEL_PATH=./models/chronos-2
export CHRONOS_DEVICE=cpu
export CHRONOS_SYMBOLS=RB0,SA0,FG0,CU0
export CHRONOS_COMMISSION=0.0002
export CHRONOS_SLIPPAGE=0.0003
```

确保 `CHRONOS_MODEL_PATH` 指向可用的 Chronos 模型目录。

---

## 3. 配置说明（config.py）

关键参数位于 `chronos_cta_v3/config.py`：

- `SYMBOLS`：要交易/研究的品种代码列表（例如 `RB0`, `CU0`）。
- `MODEL_PATH`：Chronos 模型路径（支持环境变量 `CHRONOS_MODEL_PATH` 覆盖）。
- `DEVICE`：`cpu` 或 `cuda`。
- `PRED_LEN`：预测步长。
- `TARGET_VOL`：目标波动率（用于波动率目标仓位）。
- `MAX_POSITION`：单品种最大仓位。
- `MAX_PORTFOLIO_RISK`：组合风险上限。
- `CAPITAL`：账户资金规模（用于计算名义资金）。
- `LOOKBACK` / `TRAIN_WINDOW` / `MIN_HISTORY`：历史窗口相关参数。

你可以先用默认参数跑通，再逐步调优。

---

## 4. 运行主程序

### 4.1 命令行运行（推荐统一入口）

在项目根目录执行：

```bash
python -m chronos_cta_v3.main
```

程序会输出每个可交易品种的结果（如 `alpha`、`signal`、`position`、`notional`、止损止盈价格等）。

### 4.2 在代码中调用

```python
from chronos_cta_v3.main import run

result = run()
print(result)
```

当样本不足或当期无可交易标的时，`result` 可能为空 DataFrame。

---

## 5. `backtest_engine` 回测示例

`chronos_cta_v3/backtest/backtest_engine.py` 提供了从信号到净值的完整回测工具。


### 5.1 示例一：最简回测（`backtest`）

适合你已经有：
- `returns`：每日收益率序列（如 close 的 pct_change）；
- `positions`：每日目标仓位（-1~1）。

```python
import pandas as pd
from chronos_cta_v3.backtest.backtest_engine import backtest

idx = pd.date_range("2024-01-01", periods=6, freq="D")
returns = pd.Series([0.01, -0.005, 0.003, 0.0, 0.004, -0.002], index=idx)
positions = pd.Series([0, 1, 1, -1, -1, 0], index=idx)

equity = backtest(returns, positions)
print(equity)
```

注意：引擎内部使用 `positions.shift(1)`，即 **下一根K线才生效**，避免未来函数。


### 5.2 示例二：根据预测值自动生成仓位 + 阈值优化

```python
import pandas as pd
from chronos_cta_v3.backtest.backtest_engine import auto_generate_strategy, optimize_signal_threshold

idx = pd.date_range("2024-01-01", periods=8, freq="D")
pred = pd.Series([0.01, 0.02, -0.01, -0.03, 0.005, 0.006, -0.007, 0.0], index=idx)
ret = pd.Series([0.002, 0.004, -0.003, -0.005, 0.001, 0.003, -0.002, 0.0], index=idx)

best_thr = optimize_signal_threshold(pred, ret, grid=[0.0, 0.005, 0.01])
positions = auto_generate_strategy(pred, threshold=best_thr)

print("best threshold:", best_thr)
print(positions)
```

`optimize_signal_threshold` 会在给定网格上以 Sharpe 近似指标选最优阈值。


### 5.3 示例三：滚动窗口 Walk-Forward 回测（推荐）

```python
import numpy as np
import pandas as pd
from chronos_cta_v3.backtest.backtest_engine import walk_forward_backtest

idx = pd.date_range("2023-01-01", periods=260, freq="B")
np.random.seed(42)

pred = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
ret = pd.Series(np.random.normal(0, 0.008, len(idx)), index=idx)

daily = walk_forward_backtest(
    predictions=pred,
    returns=ret,
    train_window=120,
    test_window=20,
    start_date="2023-06-01",
)

print(daily.tail())
print("final equity:", daily["equity"].iloc[-1] if not daily.empty else None)
```

输出 `daily` 主要字段：
- `date`：交易日
- `prediction`：预测值
- `position`：当日目标仓位
- `return`：标的收益率
- `pnl`：策略日收益率
- `threshold`：该滚动窗口优化得到的阈值
- `equity`：累计净值


### 5.4 示例四：使用 `BacktestPlatform` + 生成交易记录

```python
import numpy as np
import pandas as pd
from chronos_cta_v3.backtest.backtest_engine import BacktestPlatform

idx = pd.date_range("2024-01-01", periods=200, freq="B")
np.random.seed(7)

pred = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
ret = pd.Series(np.random.normal(0, 0.009, len(idx)), index=idx)
price = pd.Series(100 + np.cumsum(np.random.normal(0, 1, len(idx))), index=idx)

bt = BacktestPlatform(train_window=100, test_window=10)
daily, trades = bt.run_with_trade_records(
    predictions=pred,
    returns=ret,
    prices=price,
    start_date="2024-03-01",
)

print("daily columns:", daily.columns.tolist())
print("trade records:")
print(trades.head(10))
```

`trades` 固定字段顺序：
- `date`, `action`, `from_position`, `to_position`, `prediction`, `price`, `exec_price`, `price_slippage`, `transaction_cost`, `pnl`, `equity`

其中：
- `action`：`OPEN` / `CLOSE` / `REVERSE` / `ADJUST`
- `from_position`、`to_position`：仓位变化
- `prediction`、`price`：触发当时的预测和价格
- `pnl`、`equity`：当日收益与净值

---

## 6. 如何把主程序输出接入回测

实际使用时，建议你先让研究流程产出连续时间序列：
1. 每日预测值 `predictions`；
2. 对应标的每日收益率 `returns`（例如 `close.pct_change()`）；
3. （可选）每日收盘价 `prices` 用于交易记录展示。

然后直接调用：

```python
from chronos_cta_v3.backtest.backtest_engine import BacktestPlatform

bt = BacktestPlatform(train_window=120, test_window=20)
daily, trades = bt.run_with_trade_records(predictions, returns, prices)
```

如果你先只做单次静态阈值测试，也可以：
- `best_thr = optimize_signal_threshold(predictions, returns)`
- `positions = auto_generate_strategy(predictions, best_thr)`
- `equity = backtest(returns, positions)`

---

## 7. 常见问题（FAQ）

### Q1：运行 `main.py` 报模型路径错误
检查 `MODEL_PATH` 是否存在可加载模型，且 `DEVICE` 与本机环境匹配。

### Q2：AkShare 拉取数据失败
通常是网络或数据源临时波动。可重试，或先将数据缓存到本地后回测。

### Q3：回测结果为空
常见原因：
- `predictions` 与 `returns` 时间索引未对齐；
- 样本长度小于 `train_window`；
- `start_date` 截断后数据不足。

### Q4：为什么收益与仓位有一日偏移
回测中默认 `positions.shift(1)`，表示今日信号在下一交易日执行，这是更保守、也更接近实盘的设定。

---

## 8. 建议的最小可复现实验流程

1. 跑通最简回测示例（5.1）；
2. 跑通 walk-forward 示例（5.3）；
3. 将你自己的预测序列接入 `BacktestPlatform`（5.4）；
4. 再逐步调优 `train_window / test_window / threshold grid / 风险参数`。

这样可以先保证流程正确，再优化策略表现。
