# 因子扩展使用说明（时序 + 横截面）

本文档说明如何在 **不新增数据源**（仅使用 `open/high/low/close/volume/hold`）的前提下，使用项目内新增的因子能力，包括：

1. 时序因子（单合约）
2. 横截面因子（跨品种）
3. 行业内相对强弱（组内排名）
4. 在研究与训练流程中的推荐接入方式

---

## 1. 可用输入字段与约束

新增实现严格基于以下字段：

- 必需：`date`, `open`, `high`, `low`, `close`
- 可选（但建议提供）：`volume`, `hold`
- 横截面因子额外需要：`symbol`

说明：
- 当 `volume` 或 `hold` 缺失时，相关因子会自动退化为 `NaN`（不报错）。
- 当数据中只有单一 `symbol` 时，横截面因子不会计算（因为无“截面”可排名）。

---

## 2. 新增时序因子（A 部分）

新增因子已集成在 `compute_factors(df)` 中，按类别如下。

### 2.1 波动率分解因子

- `up_var_share_20 = upside_vol20 / (upside_vol20 + downside_vol20)`
- `jump_ratio_20 = overnight_vol_20 / vol20`
- `range_to_close_vol_20 = std((high-low)/close, 20)`

用途：
- 区分“上涨波动主导”与“下跌波动主导”；
- 衡量夜盘/隔夜跳空对总波动的贡献；
- 识别“日内振幅驱动”的波动环境。

### 2.2 趋势质量因子

- `breakout_persist_5`：近 5 天收盘价维持在 `rolling_high_60` 附近（阈值 99.5%）
- `trend_r2_20`：`log(close)` 做 20 日线性回归的 `R²`
- `slope_over_noise_20`：回归斜率 / 残差标准差

用途：
- 过滤“假突破”（只破一天就回落）；
- 定量刻画“趋势是否平滑可信”。

### 2.3 量价/持仓交互因子

- `price_volume_corr_20 = corr(ret1, vol_chg_5, 20)`
- `signed_volume_20 = sum(sign(ret1)*volume,20)/sum(volume,20)`
- `oi_price_corr_20 = corr(ret1, oi_chg_1,20)`

用途：
- 区分“价升量增”的健康趋势与“价升量缩”的弱趋势；
- 判断持仓变化与价格是否同向共振。

### 2.4 K 线结构因子

- `upper_shadow_ratio = (high-max(open,close))/(high-low)`
- `lower_shadow_ratio = (min(open,close)-low)/(high-low)`
- `body_ratio = abs(close-open)/(high-low)`
- 以及各自的 `20` 日均值与标准差：
  - `upper_shadow_ratio_mean20/std20`
  - `lower_shadow_ratio_mean20/std20`
  - `body_ratio_mean20/std20`

用途：
- 捕捉“冲高回落/探底回升/实体强弱”等微观行为；
- 与趋势类因子组合时，常用于反转与延续的分层判断。

### 2.5 分位位置（状态机）因子

- `close_rank_20 = pct_rank(close, 20)`
- `vol_rank_60 = pct_rank(vol20, 60)`
- `oi_rank_60 = pct_rank(hold, 60)`

用途：
- 将连续变量映射为 [0,1] 的相对状态，增强树模型鲁棒性；
- 降低跨品种尺度差异影响。

---

## 3. 新增横截面因子（B 部分）

新增函数：`add_cross_sectional_factors(df)`，输入为 **多品种面板数据**（同表含 `date + symbol`）。

### 3.1 横截面动量/反转

- `cs_mom_rank`：按交易日对 `mom20` 做截面分位排名
- `cs_reversal_5`：按交易日对 `mom5` 做截面排名（短反转视角）

### 3.2 横截面波动/拥挤度

- `cs_vol_rank`：按交易日对 `vol20` 做截面分位排名
- `cs_oi_chg_rank`：按交易日对 `oi_chg_20` 做截面分位排名
- `crowding_score = z(ret20)+z(oi_chg_20)+z(volume_ratio_20_60)`
- `crowding_cs_rank`：拥挤度的截面分位排名

其中 `z(*)` 在实现中按 **symbol 内 60 日滚动标准化**，再做当日截面比较。

### 3.3 行业内相对强弱

实现内置了期货品种到行业组映射（黑色/有色/化工/农产品），并生成：

- `industry_group`
- `industry_mom_rank`：按 `date + industry_group` 对 `mom20` 组内排名

这可以减少“大类风险偏置”对因子解释的干扰。

---

## 4. 推荐接入范式

### 4.1 单品种时序研究（现有主流程兼容）

```python
from chronos_cta_v3.factors.factor_library import compute_factors

feat = compute_factors(single_symbol_df)
```

说明：  
这会自动得到全部新增时序因子；横截面因子在单品种下会被跳过（返回为空列或不生成）。

### 4.2 多品种横截面研究（推荐新流程）

```python
import pandas as pd
from chronos_cta_v3.factors.factor_library import compute_factors, add_cross_sectional_factors

# panel_raw: 包含多品种的原始 OHLCV+hold，至少含 date/symbol 列
panel_feat = (
    panel_raw
    .sort_values(["symbol", "date"])
    .groupby("symbol", group_keys=False)
    .apply(compute_factors)
    .reset_index(drop=True)
)

panel_feat = add_cross_sectional_factors(panel_feat)
```

建议：
1. 先按 `symbol` 计算时序因子；
2. 再统一调用横截面因子函数；
3. 训练时可混合使用时序 + 截面特征。

---

## 5. 建模实践建议

1. **避免信息泄露**：  
   目标列（例如 `t+1` 收益）在组装训练集时必须 `.shift(-1)`，并只用当前与历史因子。

2. **缺失值处理**：  
   新增滚动类因子在前若干日会出现 `NaN`，建议：
   - 训练前统一 `dropna(subset=核心特征)`；
   - 或使用模型可解释且稳定的填充策略。

3. **特征分层**：  
   可以将特征分为：
   - 方向（mom/trend）
   - 风险（vol/tail/jump）
   - 拥挤（oi/volume/crowding）
   - 微观结构（shadow/body）
   再进行分组 IC、分组重要性评估。

4. **横截面标签设计**（进阶）：  
   若改为日度截面模型，可将标签改为当日全品种的未来收益 rank / zscore，以提升截面稳定性。

---

## 6. 快速检查清单

- [ ] 单品种 DataFrame 是否包含 `date/open/high/low/close`  
- [ ] 横截面 DataFrame 是否额外包含 `symbol`  
- [ ] 是否先按 `symbol` 排序再计算滚动因子  
- [ ] 是否在训练集构建时做了 target shift 与样本对齐  
- [ ] 是否对前置 `NaN` 做了统一处理

---

## 7. 相关代码位置

- 因子主实现：`chronos_cta_v3/factors/factor_library.py`
- 单元测试：`tests/test_factor_expansion.py`

