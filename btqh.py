import backtrader as bt
import akshare as ak
import pandas as pd
import warnings
from datetime import timedelta, datetime
warnings.filterwarnings('ignore')

# ===================== 1. 读取Excel信号数据 =====================
def load_signal_data(excel_path):
    """读取Excel中的交易信号、止损止盈、日期等数据，按品种+日期整理"""
    df = pd.read_excel(excel_path)
    # 数据清洗：转换日期（保留datetime类型）、过滤有效信号（signal=1/-1）
    df['date'] = pd.to_datetime(df['date'])  # 保留datetime.datetime类型，不转date
    df = df[df['signal'].isin([1, -1])]  # 仅保留开仓信号（1多头/-1空头）
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 按品种分组，构建信号字典：{symbol: {date: {'signal':, 'stop_loss':, 'target':}}}
    signal_dict = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        signal_dict[symbol] = {}
        for _, row in symbol_df.iterrows():
            # trade_date = 信号日期+1天（保持datetime类型）
            trade_date = row['date'] + timedelta(days=1)
            signal_dict[symbol][trade_date.date()] = {  # 用date()作为key，方便后续匹配
                'signal': row['signal'],
                'stop_loss_price': row['stop_loss_price'],
                'target_price': row['target_price']
            }
    return signal_dict

# ===================== 2. 期货数据获取函数（兼容信号数据合并） =====================
def get_futures_data(symbol, start_date="2024-01-01", end_date="2026-03-20"):
    """获取期货K线数据，返回标准化DataFrame"""
    try:
        print(f"正在获取 {symbol} 数据...")
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df.empty:
            print(f"{symbol} 数据源返回空")
            return pd.DataFrame()
        print(f"{symbol} 数据获取成功，记录数：{len(df)}")
        
        # 数据清洗：日期保留datetime.datetime类型（关键修复点）
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # 不转dt.date
        df = df.dropna(subset=['date'])
        # 日期过滤（转为datetime后对比）
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        df = df.sort_values('date').reset_index(drop=True)
        
        # 检查必要列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"{symbol} 缺少必要列：{set(required_cols) - set(df.columns)}")
            return pd.DataFrame()
        
        # 缺失值填充
        df[required_cols] = df[required_cols].fillna(method='ffill').fillna(0)
        return df
    except Exception as e:
        print(f"获取{symbol}数据失败：{e}")
        return pd.DataFrame()

# ===================== 3. 自定义多空风险平价策略（修复closeorder+datetime方法调用错误） =====================
class SignalBasedFuturesStrategy(bt.Strategy):
    """
    基于Excel信号的交易策略：
    1. 信号日期+1天执行交易（date→date+1）
    2. signal=1→开多头，signal=-1→开空头
    3. 已有同方向仓位时，同信号不加仓，仅维持
    4. 未到止损/止盈前，持续持仓
    5. 新增：完整记录每笔交易信息（开仓/平仓、价格、数量、盈亏、止损止盈等）
    """
    params = (
        ('initial_capital', 100000),
        ('commission_rate', 0.0003),  # 万3佣金
        ('slip_point', 0.2),          # 固定滑点
        ('risk_free_rate', 0.02),     # 无风险利率
        ('excel_path', 'chronos_cta_v3_backtest_results.xlsx'),  # Excel信号文件路径
    )

    def __init__(self):
        # 加载Excel信号数据
        self.signal_dict = load_signal_data(self.params.excel_path)
        # 跟踪每个品种的状态：扩展字段记录止损/止盈价
        self.symbol_status = {}
        for symbol in self.signal_dict.keys():
            self.symbol_status[symbol] = {
                'position': 0,  # 0:无仓位, 1:多头, -1:空头
                'stop_order': None,  # 止损订单
                'target_order': None,  # 止盈订单
                'stop_loss_price': None,  # 记录止损价
                'target_price': None,    # 记录止盈价
                'current_trade_close_type': None  # 新增：记录当前交易的平仓类型
            }
        # 映射数据源与品种名（确保datas和signal_dict匹配）
        self.data_symbol_map = {d._name: d for d in self.datas if d._name in self.signal_dict}
        # ========== 新增：初始化交易记录列表 ==========
        self.trade_records = []

    def next(self):
        """每个K线周期执行交易逻辑"""
        # 获取当前回测日期（转为date对象，匹配signal_dict的key）
        current_datetime = self.datas[0].datetime.datetime(0)
        current_date = current_datetime.date()
        
        for symbol, data in self.data_symbol_map.items():
            status = self.symbol_status[symbol]
            # 检查当前日期是否有对应交易信号
            if current_date not in self.signal_dict[symbol]:
                continue  # 无信号则跳过
            
            # 获取当前日期的信号参数
            signal_info = self.signal_dict[symbol][current_date]
            signal = signal_info['signal']
            stop_loss = signal_info['stop_loss_price']
            target_price = signal_info['target_price']
            current_close = data.close[0]  # 当前品种收盘价

            # 1. 无仓位时，执行开仓
            if status['position'] == 0:
                # 计算仓位（按风险平价权重，这里简化为固定比例）
                total_cash = self.broker.getcash()
                # 从Excel中获取该品种的risk_parity_weight
                risk_weight = self.get_risk_parity_weight(symbol, current_date - timedelta(days=1))
                position_cash = total_cash * risk_weight
                position_size = round(position_cash / current_close, 2)  # 取整
                
                if position_size <= 0:
                    continue

                # 多头开仓（signal=1）
                if signal == 1:
                    self.buy(data=data, size=position_size, exectype=bt.Order.Market)
                    print(f"【开仓】{symbol} | 日期：{current_date} | 方向：多头 | 价格：{current_close} | 数量：{position_size}")
                    # ========== 新增：记录止损/止盈价 ==========
                    self.symbol_status[symbol]['stop_loss_price'] = stop_loss
                    self.symbol_status[symbol]['target_price'] = target_price
                    # 挂止损单（跌破止损价平仓）
                    stop_order = self.sell(data=data, size=position_size, price=stop_loss, exectype=bt.Order.Stop, close=True)
                    # 挂止盈单（达到目标价平仓）
                    target_order = self.sell(data=data, size=position_size, price=target_price, exectype=bt.Order.Limit, close=True)
                    status['position'] = 1
                    status['stop_order'] = stop_order
                    status['target_order'] = target_order
                    # 重置平仓类型记录
                    status['current_trade_close_type'] = None

                # 空头开仓（signal=-1）
                elif signal == -1:
                    self.sell(data=data, size=position_size, exectype=bt.Order.Market)
                    print(f"【开仓】{symbol} | 日期：{current_date} | 方向：空头 | 价格：{current_close} | 数量：{position_size}")
                    # ========== 新增：记录止损/止盈价 ==========
                    self.symbol_status[symbol]['stop_loss_price'] = stop_loss
                    self.symbol_status[symbol]['target_price'] = target_price
                    # 挂止损单（突破止损价平仓）
                    stop_order = self.buy(data=data, size=position_size, price=stop_loss, exectype=bt.Order.Stop, close=True)
                    # 挂止盈单（达到目标价平仓）
                    target_order = self.buy(data=data, size=position_size, price=target_price, exectype=bt.Order.Limit, close=True)
                    status['position'] = -1
                    status['stop_order'] = stop_order
                    status['target_order'] = target_order
                    # 重置平仓类型记录
                    status['current_trade_close_type'] = None

            # 2. 已有同方向仓位时，维持持仓（不加仓）
            elif status['position'] == signal:
                print(f"【持仓】{symbol} | 日期：{current_date} | 方向：{'多头' if signal==1 else '空头'} | 维持仓位，不加仓")

            # 3. 仓位方向与当前信号相反（暂不平仓）
            else:
                print(f"【忽略】{symbol} | 日期：{current_date} | 当前仓位：{'多头' if status['position']==1 else '空头'} | 信号方向相反，暂不平仓")

    def get_risk_parity_weight(self, symbol, signal_date):
        """从Excel中获取对应品种+信号日期的risk_parity_weight"""
        # 重新读取Excel（简化逻辑，可优化为初始化时缓存）
        df = pd.read_excel(self.params.excel_path)
        df['date'] = pd.to_datetime(df['date']).dt.date  # 转为date对象匹配
        # 筛选条件：品种匹配 + 信号日期匹配
        filter_df = df[(df['symbol'] == symbol) & (df['date'] == signal_date)]
        if not filter_df.empty:
            return filter_df.iloc[0]['risk_parity_weight']
        return 0.05  # 默认权重（无数据时）

    def notify_order(self, order):
        """订单状态通知：更新订单状态 + 标记平仓类型"""
        # 仅处理已完成/取消/保证金不足的订单
        if order.status not in [order.Completed, order.Canceled, order.Margin]:
            return
            
        symbol = order.data._name if order.data else None
        # 安全校验：确保symbol有效且在status字典中
        if not symbol or symbol not in self.symbol_status:
            return
        status = self.symbol_status[symbol]
        
        # 止损/止盈订单完成后，标记平仓类型并重置仓位状态
        if order == status['stop_order'] and order.status == order.Completed:
            status['current_trade_close_type'] = "止损"
            # 重置状态
            status['position'] = 0
            status['stop_order'] = None
            status['target_order'] = None
            current_date = self.datas[0].datetime.date(0)
            print(f"【平仓】{symbol} | 日期：{current_date} | 类型：止损 | 价格：{order.executed.price}")
        elif order == status['target_order'] and order.status == order.Completed:
            status['current_trade_close_type'] = "止盈"
            # 重置状态
            status['position'] = 0
            status['stop_order'] = None
            status['target_order'] = None
            current_date = self.datas[0].datetime.date(0)
            print(f"【平仓】{symbol} | 日期：{current_date} | 类型：止盈 | 价格：{order.executed.price}")
        # 修正：判断平仓订单的正确方式（使用order.params.close）
        elif status['position'] != 0 and hasattr(order.params, 'close') and order.params.close and order.status == order.Completed:
            status['current_trade_close_type'] = "手动平仓"
            status['position'] = 0
            current_date = self.datas[0].datetime.date(0)
            print(f"【平仓】{symbol} | 日期：{current_date} | 类型：手动平仓 | 价格：{order.executed.price}")

    def notify_trade(self, trade):
        """交易完成通知：记录完整交易信息（修复Trade.executed属性错误）"""
        if trade.isclosed:
            symbol = trade.data._name
            pnl_net = trade.pnlcomm
            current_date = self.datas[0].datetime.date(0)
            current_datetime = datetime.combine(current_date, datetime.now().time())  # 统一缺省datetime
            
            # ========== 修复：从开仓/平仓订单获取成交时间（修复datetime方法调用） ==========
            # 开仓时间/日期：从开仓订单的executed.dt获取
            open_dt = trade.dtopen if (trade.dtopen and hasattr(trade.dtopen, 'dt')) else current_datetime
            open_date = open_dt.date()  # 纯date类型
            open_time = open_dt.time()  # 纯time类型
            
            # 平仓时间/日期：从平仓订单的executed.dt获取
            close_dt = trade.dtclose if (trade.dtclose and hasattr(trade.dtclose, 'dt')) else current_datetime
            close_date = close_dt.date()
            close_time = close_dt.time()
            
            # ========== 核心修复：从开仓/平仓订单获取成交价格 ==========
            # 开仓价格：从开仓订单的executed.price获取
            open_price = 0.0
            if trade.dtopen and hasattr(trade.dtopen, 'executed') and trade.dtopen.executed.price:
                open_price = round(trade.dtopen, 2)

            # 平仓价格：从平仓订单的executed.price获取
            close_price = 0.0
            if trade.dtclose:
                close_price = round(trade.dtclose, 2)

            # 确定交易方向
            direction = "多头" if trade.size > 0 else "空头"
            
            # 组装交易记录字段（类型完全统一）
            trade_record = {
                "品种代码": symbol,
                "开仓日期": open_date,          # 纯date类型
                "开仓时间": open_time,          # 纯time类型
                "开仓价格": open_price,         # 实际开仓成交价格
                "平仓日期": close_date,         # 纯date类型
                "平仓时间": close_time,         # 纯time类型
                "平仓价格": close_price,        # 实际平仓成交价格
                "交易方向": direction,
                "持仓数量": abs(trade.size) if trade.size else 0,
                "平仓类型": self.symbol_status[symbol].get('current_trade_close_type', '未知'),
                "毛盈亏(元)": round(trade.pnl, 2) if trade.pnl else 0,
                "手续费(元)": round(trade.commission, 2) if trade.commission else 0,
                "滑点成本(元)": round(self.params.slip_point * abs(trade.size), 2) if trade.size else 0,
                "净盈亏(元)": round(trade.pnlcomm, 2) if trade.pnlcomm else 0,
                "预设止损价": self.symbol_status[symbol].get('stop_loss_price'),
                "预设止盈价": self.symbol_status[symbol].get('target_price'),
                "开仓订单类型": "市价单",
                "平仓订单类型": "止损单" if self.symbol_status[symbol].get('current_trade_close_type') == "止损" 
                            else "止盈单" if self.symbol_status[symbol].get('current_trade_close_type') == "止盈" 
                            else "市价单"
            }
            
            # 添加到交易记录列表
            self.trade_records.append(trade_record)
            # 重置当前交易的平仓类型
            self.symbol_status[symbol]['current_trade_close_type'] = None

# ===================== 4. 回测主程序（新增保存Excel） =====================
if __name__ == '__main__':
    # 初始化回测引擎
    cerebro = bt.Cerebro()

    # 1. 加载Excel信号数据，获取所有需回测的品种
    excel_path = f'E:/股票数据/chronos_cta_v3_backtest_results.xlsx'
    signal_dict = load_signal_data(excel_path)
    target_symbols = list(signal_dict.keys())  # 需回测的品种列表

    # 2. 加载每个品种的K线数据（Akshare）
    for symbol in target_symbols:
        df = get_futures_data(
            symbol=symbol,
            start_date="2024-01-01",
            end_date="2026-03-20"
        )
        if df.empty:
            print(f"⚠️ {symbol} 无有效K线数据，跳过")
            continue
        
        # 转换为Backtrader数据源
        data_feed = bt.feeds.PandasData(
            dataname=df,
            datetime='date',
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1  # 无持仓量数据
        )
        cerebro.adddata(data_feed, name=symbol)

    # 3. 配置回测参数
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0003)
    cerebro.broker.set_slippage_fixed(0.2)

    # 4. 自定义资产总值分析器
    class TotalValueAnalyzer(bt.Analyzer):
        def __init__(self):
            self.value_history = []
        def next(self):
            self.value_history.append({
                'date': self.strategy.datas[0].datetime.date(0),
                'total_value': self.strategy.broker.getvalue()
            })
        def get_analysis(self):
            return {
                'final_value': self.value_history[-1]['total_value'] if self.value_history else 0,
                'value_history': self.value_history
            }

    # 5. 添加策略和分析器
    cerebro.addstrategy(SignalBasedFuturesStrategy, excel_path=excel_path)
    cerebro.addanalyzer(TotalValueAnalyzer, _name='total_value')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, 
                        _name='sharpe', 
                        riskfreerate=0.02,
                        annualize=True,
                        timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # 6. 运行回测
    print("="*60)
    print("开始回测...")
    print("="*60)
    results = cerebro.run()
    if not results:
        print("⚠️ 回测无结果（无有效数据源）")
    else:
        strat = results[0]
        # 提取核心回测结果
        initial_value = 100000.0
        final_total_value = strat.analyzers.total_value.get_analysis()['final_value']
        total_return = (final_total_value - initial_value) / initial_value * 100

        # 夏普比率
        try:
            sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio']
            sharpe_ratio = round(sharpe_ratio, 4)
        except Exception as e:
            sharpe_ratio = "数据不足"
            print(f"⚠️ 夏普比率计算提示：{e}")

        # 输出核心结果
        print("\n" + "="*60)
        print("回测核心结果")
        print("="*60)
        print(f"初始资金：{initial_value:.2f} 元")
        print(f"最终资产总值：{final_total_value:.2f} 元")
        print(f"总收益率：{total_return:.2f}%")
        print(f"年化夏普比率：{sharpe_ratio}")
        print("="*60)

        # ========== 保存交易记录到Excel ==========
        if strat.trade_records:
            # 转换为DataFrame并排序
            trade_df = pd.DataFrame(strat.trade_records)
            trade_df = trade_df.sort_values(by="开仓日期").reset_index(drop=True)
            
            # 保存路径（可自定义）
            trade_excel_path = "E:/股票数据/期货策略交易记录.xlsx"
            try:
                # 保存Excel（需安装openpyxl）
                trade_df.to_excel(trade_excel_path, index=False, engine='openpyxl')
                print(f"\n✅ 交易记录已成功保存！")
                print(f"📁 文件路径：{trade_excel_path}")
                print(f"📊 共记录 {len(trade_df)} 笔交易")
                # 预览前5条记录
                print("\n📌 交易记录预览（前5条）：")
                print(trade_df.head())
            except ImportError:
                print("\n❌ 保存Excel失败：请先安装openpyxl（执行：pip install openpyxl）")
            except Exception as e:
                print(f"\n❌ 保存交易记录失败：{e}")
        else:
            print("\n⚠️ 本次回测无交易记录可保存")

    # 可选：绘图
    try:
        cerebro.plot(style='candlestick', iplot=False)
    except Exception as e:
        print(f"⚠️ 绘图失败：{e}（需安装matplotlib）")