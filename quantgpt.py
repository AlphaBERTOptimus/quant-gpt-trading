import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import requests
import json
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="QuantGPT Pro - AI量化交易平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .strategy-description {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 策略类定义
class StrategyEngine:
    def __init__(self):
        self.strategies = {
            "趋势跟踪": self.trend_following,
            "均值回归": self.mean_reversion,
            "动量策略": self.momentum_strategy,
            "配对交易": self.pairs_trading,
            "突破策略": self.breakout_strategy,
            "网格交易": self.grid_trading
        }
    
    def trend_following(self, data, params):
        """趋势跟踪策略"""
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
        data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
        
        # 生成信号
        data['Signal'] = 0
        data['Signal'][short_window:] = np.where(
            data['SMA_short'][short_window:] > data['SMA_long'][short_window:], 1, 0
        )
        data['Position'] = data['Signal'].diff()
        
        return data, f"当短期均线({short_window}日)上穿长期均线({long_window}日)时买入，下穿时卖出"
    
    def mean_reversion(self, data, params):
        """均值回归策略"""
        window = params.get('window', 20)
        std_dev = params.get('std_dev', 2)
        
        data['SMA'] = data['Close'].rolling(window=window).mean()
        data['STD'] = data['Close'].rolling(window=window).std()
        data['Upper'] = data['SMA'] + (data['STD'] * std_dev)
        data['Lower'] = data['SMA'] - (data['STD'] * std_dev)
        
        data['Signal'] = 0
        data['Signal'] = np.where(data['Close'] < data['Lower'], 1, 
                                np.where(data['Close'] > data['Upper'], -1, 0))
        data['Position'] = data['Signal'].diff()
        
        return data, f"当价格跌破下轨({std_dev}倍标准差)时买入，涨破上轨时卖出"
    
    def momentum_strategy(self, data, params):
        """动量策略"""
        window = params.get('window', 14)
        threshold = params.get('threshold', 70)
        
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
        
        # 组合信号
        data['Signal'] = 0
        data['Signal'] = np.where(
            (data['RSI'] > threshold) & (data['MACD'] > data['MACD_signal']), 1,
            np.where((data['RSI'] < (100-threshold)) & (data['MACD'] < data['MACD_signal']), -1, 0)
        )
        data['Position'] = data['Signal'].diff()
        
        return data, f"基于RSI({window}日)和MACD指标的动量组合策略"
    
    def pairs_trading(self, data, params):
        """配对交易策略（简化版）"""
        window = params.get('window', 30)
        
        # 这里简化为单一资产的配对交易逻辑
        data['MA'] = data['Close'].rolling(window=window).mean()
        data['Spread'] = data['Close'] - data['MA']
        data['Spread_MA'] = data['Spread'].rolling(window=window).mean()
        data['Spread_STD'] = data['Spread'].rolling(window=window).std()
        
        data['Signal'] = np.where(
            data['Spread'] > data['Spread_MA'] + data['Spread_STD'], -1,
            np.where(data['Spread'] < data['Spread_MA'] - data['Spread_STD'], 1, 0)
        )
        data['Position'] = data['Signal'].diff()
        
        return data, f"基于价格与{window}日均值差价的配对交易策略"
    
    def breakout_strategy(self, data, params):
        """突破策略"""
        window = params.get('window', 20)
        
        data['High_max'] = data['High'].rolling(window=window).max()
        data['Low_min'] = data['Low'].rolling(window=window).min()
        
        data['Signal'] = 0
        data['Signal'] = np.where(
            data['Close'] > data['High_max'].shift(1), 1,
            np.where(data['Close'] < data['Low_min'].shift(1), -1, 0)
        )
        data['Position'] = data['Signal'].diff()
        
        return data, f"突破{window}日最高点买入，跌破{window}日最低点卖出"
    
    def grid_trading(self, data, params):
        """网格交易策略"""
        grid_size = params.get('grid_size', 0.02)  # 2%网格
        
        data['Price_change'] = data['Close'].pct_change()
        data['Cumulative_change'] = data['Price_change'].cumsum()
        
        # 简化的网格逻辑
        data['Grid_level'] = (data['Cumulative_change'] / grid_size).round()
        data['Signal'] = -data['Grid_level'].diff()  # 价格上涨卖出，下跌买入
        data['Position'] = data['Signal'].diff()
        
        return data, f"基于{grid_size*100:.1f}%网格间距的网格交易策略"

# 风险管理模块
class RiskManager:
    @staticmethod
    def add_stop_loss(data, stop_loss_pct=0.05):
        """添加止损"""
        data['Stop_Loss'] = np.nan
        data['Stop_Loss_Signal'] = 0
        
        entry_price = None
        position = 0
        
        for i in range(len(data)):
            if data['Position'].iloc[i] == 1:  # 买入信号
                entry_price = data['Close'].iloc[i]
                position = 1
            elif data['Position'].iloc[i] == -1:  # 卖出信号
                entry_price = None
                position = 0
            
            if position == 1 and entry_price:
                stop_price = entry_price * (1 - stop_loss_pct)
                data.iloc[i, data.columns.get_loc('Stop_Loss')] = stop_price
                
                if data['Close'].iloc[i] < stop_price:
                    data.iloc[i, data.columns.get_loc('Stop_Loss_Signal')] = -1
                    position = 0
                    entry_price = None
        
        return data
    
    @staticmethod
    def calculate_position_size(data, risk_per_trade=0.02, account_size=100000):
        """计算仓位大小"""
        data['Position_Size'] = account_size * risk_per_trade / data['Close']
        return data
    
    @staticmethod
    def calculate_var(returns, confidence=0.05):
        """计算VaR"""
        return np.percentile(returns, confidence * 100)

# 技术指标库
class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(data):
        """添加所有技术指标"""
        # 趋势指标
        data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
        data['MACD_hist'] = ta.trend.MACD(data['Close']).macd_diff()
        
        # 动量指标
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['Stoch'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
        data['Williams_R'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
        
        # 波动率指标
        data['BB_upper'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
        data['BB_middle'] = ta.volatility.BollingerBands(data['Close']).bollinger_mavg()
        data['BB_lower'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        
        # 成交量指标
        if 'Volume' in data.columns:
            data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
            data['Volume_SMA'] = ta.volume.VolumeSMAIndicator(data['Close'], data['Volume']).volume_sma()
        
        return data

# 数据获取模块
class DataManager:
    @staticmethod
    def get_yahoo_data(symbol, period="2y"):
        """获取Yahoo Finance数据"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"获取数据失败: {e}")
            return None
    
    @staticmethod
    def get_alpha_vantage_data(symbol, api_key):
        """获取Alpha Vantage数据"""
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
            response = requests.get(url)
            data = response.json()
            
            if "Time Series (Daily)" in data:
                df = pd.DataFrame(data["Time Series (Daily)"]).T
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return df.sort_index()
            else:
                st.error("Alpha Vantage API返回错误")
                return None
        except Exception as e:
            st.error(f"Alpha Vantage数据获取失败: {e}")
            return None
    
    @staticmethod
    def get_news_sentiment(symbol):
        """获取新闻情感数据（模拟）"""
        # 这里是模拟数据，实际应用中可以接入真实的新闻API
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        sentiment_scores = np.random.normal(0, 0.3, 30)  # 模拟情感分数
        
        return pd.DataFrame({
            'Date': dates,
            'Sentiment': sentiment_scores,
            'News_Count': np.random.randint(5, 50, 30)
        })

# 回测引擎
class BacktestEngine:
    @staticmethod
    def run_backtest(data, initial_capital=100000, commission=0.001):
        """运行回测"""
        data = data.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
        
        # 考虑手续费
        data['Strategy_Returns'] = data['Strategy_Returns'] - (np.abs(data['Position']) * commission)
        
        # 计算累计收益
        data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
        data['Strategy_Cumulative'] = (1 + data['Strategy_Returns']).cumprod()
        
        # 计算最终资产
        data['Portfolio_Value'] = initial_capital * data['Strategy_Cumulative']
        
        return data
    
    @staticmethod
    def calculate_metrics(data):
        """计算回测指标"""
        strategy_returns = data['Strategy_Returns'].dropna()
        
        # 基本指标
        total_return = data['Strategy_Cumulative'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # 最大回撤
        cumulative = data['Strategy_Cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        winning_trades = len(strategy_returns[strategy_returns > 0])
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # VaR
        var_95 = RiskManager.calculate_var(strategy_returns, 0.05)
        
        return {
            "总收益率": f"{total_return:.2%}",
            "年化收益率": f"{annual_return:.2%}",
            "年化波动率": f"{volatility:.2%}",
            "夏普比率": f"{sharpe_ratio:.2f}",
            "最大回撤": f"{max_drawdown:.2%}",
            "胜率": f"{win_rate:.2%}",
            "VaR(95%)": f"{var_95:.2%}",
            "交易次数": total_trades
        }

# 参数优化模块
class ParameterOptimizer:
    @staticmethod
    def grid_search(data, strategy_func, param_grid, metric='sharpe_ratio'):
        """网格搜索优化"""
        best_params = None
        best_score = -np.inf
        results = []
        
        for params in ParameterGrid(param_grid):
            try:
                # 运行策略
                test_data, _ = strategy_func(data.copy(), params)
                test_data = BacktestEngine.run_backtest(test_data)
                
                # 计算目标指标
                strategy_returns = test_data['Strategy_Returns'].dropna()
                if len(strategy_returns) > 0:
                    if metric == 'sharpe_ratio':
                        annual_return = strategy_returns.mean() * 252
                        volatility = strategy_returns.std() * np.sqrt(252)
                        score = annual_return / volatility if volatility != 0 else 0
                    elif metric == 'total_return':
                        score = test_data['Strategy_Cumulative'].iloc[-1] - 1
                    else:
                        score = 0
                    
                    results.append({
                        'params': params,
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
            except:
                continue
        
        return best_params, best_score, results

# 主应用
def main():
    # 标题
    st.markdown('<h1 class="main-header">🚀 QuantGPT Pro - AI量化交易平台</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    st.sidebar.title("📊 策略配置")
    
    # 选择数据源
    data_source = st.sidebar.selectbox(
        "选择数据源",
        ["Yahoo Finance", "Alpha Vantage"]
    )
    
    # 股票代码输入
    symbol = st.sidebar.text_input("股票代码", value="AAPL").upper()
    
    # Alpha Vantage API Key（如果选择）
    if data_source == "Alpha Vantage":
        api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
    
    # 时间范围
    period = st.sidebar.selectbox(
        "数据时间范围",
        ["1y", "2y", "5y", "max"]
    )
    
    # 策略选择
    strategy_engine = StrategyEngine()
    selected_strategy = st.sidebar.selectbox(
        "选择策略类型",
        list(strategy_engine.strategies.keys())
    )
    
    # 策略参数设置
    st.sidebar.subheader("策略参数")
    params = {}
    
    if selected_strategy == "趋势跟踪":
        params['short_window'] = st.sidebar.slider("短期窗口", 5, 50, 20)
        params['long_window'] = st.sidebar.slider("长期窗口", 20, 200, 50)
    elif selected_strategy == "均值回归":
        params['window'] = st.sidebar.slider("移动窗口", 10, 50, 20)
        params['std_dev'] = st.sidebar.slider("标准差倍数", 1.0, 3.0, 2.0, 0.1)
    elif selected_strategy == "动量策略":
        params['window'] = st.sidebar.slider("RSI窗口", 5, 30, 14)
        params['threshold'] = st.sidebar.slider("RSI阈值", 60, 80, 70)
    elif selected_strategy == "配对交易":
        params['window'] = st.sidebar.slider("统计窗口", 20, 60, 30)
    elif selected_strategy == "突破策略":
        params['window'] = st.sidebar.slider("突破窗口", 10, 50, 20)
    elif selected_strategy == "网格交易":
        params['grid_size'] = st.sidebar.slider("网格大小(%)", 0.5, 5.0, 2.0, 0.1) / 100
    
    # 风险管理参数
    st.sidebar.subheader("风险管理")
    use_stop_loss = st.sidebar.checkbox("启用止损")
    if use_stop_loss:
        stop_loss_pct = st.sidebar.slider("止损百分比(%)", 1, 10, 5) / 100
    
    risk_per_trade = st.sidebar.slider("单笔风险(%)", 1, 5, 2) / 100
    initial_capital = st.sidebar.number_input("初始资金", value=100000, step=10000)
    
    # 参数优化选项
    st.sidebar.subheader("参数优化")
    enable_optimization = st.sidebar.checkbox("启用参数优化")
    
    if st.sidebar.button("🚀 运行回测"):
        # 获取数据
        with st.spinner("正在获取数据..."):
            if data_source == "Yahoo Finance":
                data = DataManager.get_yahoo_data(symbol, period)
            else:
                if 'api_key' in locals() and api_key:
                    data = DataManager.get_alpha_vantage_data(symbol, api_key)
                else:
                    st.error("请输入Alpha Vantage API Key")
                    return
        
        if data is not None and not data.empty:
            # 添加技术指标
            data = TechnicalIndicators.add_all_indicators(data)
            
            # 参数优化
            if enable_optimization:
                with st.spinner("正在优化参数..."):
                    if selected_strategy == "趋势跟踪":
                        param_grid = {
                            'short_window': [10, 15, 20, 25],
                            'long_window': [40, 50, 60, 70]
                        }
                    elif selected_strategy == "均值回归":
                        param_grid = {
                            'window': [15, 20, 25, 30],
                            'std_dev': [1.5, 2.0, 2.5]
                        }
                    else:
                        param_grid = [params]  # 默认参数
                    
                    if len(param_grid) > 1:
                        best_params, best_score, _ = ParameterOptimizer.grid_search(
                            data, strategy_engine.strategies[selected_strategy], param_grid
                        )
                        st.success(f"最优参数: {best_params}, 夏普比率: {best_score:.2f}")
                        params = best_params
            
            # 运行策略
            with st.spinner("正在运行策略..."):
                strategy_data, strategy_description = strategy_engine.strategies[selected_strategy](data.copy(), params)
                
                # 添加风险管理
                if use_stop_loss:
                    strategy_data = RiskManager.add_stop_loss(strategy_data, stop_loss_pct)
                
                strategy_data = RiskManager.calculate_position_size(strategy_data, risk_per_trade, initial_capital)
                
                # 运行回测
                backtest_data = BacktestEngine.run_backtest(strategy_data, initial_capital)
                
                # 计算指标
                metrics = BacktestEngine.calculate_metrics(backtest_data)
            
            # 显示结果
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 策略表现")
                
                # 策略描述
                st.markdown(f'<div class="strategy-description"><strong>策略说明:</strong> {strategy_description}</div>', 
                           unsafe_allow_html=True)
                
                # 价格和信号图表
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('价格走势与交易信号', '技术指标', '策略收益'),
                    row_width=[0.3, 0.3, 0.4]
                )
                
                # 价格线
                fig.add_trace(
                    go.Scatter(x=backtest_data.index, y=backtest_data['Close'], 
                              name='收盘价', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # 买入信号
                buy_signals = backtest_data[backtest_data['Position'] == 1]
                if not buy_signals.empty:
                    fig.add_trace(
                        go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                  mode='markers', name='买入', 
                                  marker=dict(color='green', size=8, symbol='triangle-up')),
                        row=1, col=1
                    )
                
                # 卖出信号
                sell_signals = backtest_data[backtest_data['Position'] == -1]
                if not sell_signals.empty:
                    fig.add_trace(
                        go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                  mode='markers', name='卖出',
                                  marker=dict(color='red', size=8, symbol='triangle-down')),
                        row=1, col=1
                    )
                
                # 技术指标
                if 'RSI' in backtest_data.columns:
                    fig.add_trace(
                        go.Scatter(x=backtest_data.index, y=backtest_data['RSI'],
                                  name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                
                # 策略收益
                fig.add_trace(
                    go.Scatter(x=backtest_data.index, y=(backtest_data['Cumulative_Returns']-1)*100,
                              name='买入持有收益(%)', line=dict(color='gray')),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=backtest_data.index, y=(backtest_data['Strategy_Cumulative']-1)*100,
                              name='策略收益(%)', line=dict(color='green')),
                    row=3, col=1
                )
                
                fig.update_layout(height=800, showlegend=True, title_text=f"{symbol} 策略回测结果")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 回测指标")
                
                # 指标展示
                for metric, value in metrics.items():
                    st.markdown(f'<div class="metric-card"><strong>{metric}:</strong> {value}</div>', 
                               unsafe_allow_html=True)
                    st.markdown("")  # 空行
                
                # 风险指标
                st.subheader("🛡️ 风险分析")
                strategy_returns = backtest_data['Strategy_Returns'].dropna()
                
                if len(strategy_returns) > 0:
                    fig_risk = go.Figure()
                    fig_risk.add_trace(go.Histogram(x=strategy_returns*100, name='收益分布'))
                    fig_risk.update_layout(title="策略收益分布", xaxis_title="日收益率(%)")
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                # 月度收益热力图
                st.subheader("📅 月度收益")
                monthly_returns = strategy_returns.resample('M').apply(lambda x: (1+x).prod()-1)
                if len(monthly_returns) > 0:
                    monthly_df = monthly_returns.to_frame('Returns')
                    monthly_df['Year'] = monthly_df.index.year
                    monthly_df['Month'] = monthly_df.index.month
                    pivot_table = monthly_df.pivot_table(values='Returns', index='Year', columns='Month')
                    
                    fig_heatmap = px.imshow(pivot_table.values*100, 
                                          x=[f'{i}月' for i in range(1, 13)],
                                          y=pivot_table.index,
                                          color_continuous_scale='RdYlGn',
                                          aspect='auto')
                    fig_heatmap.update_layout(title="月度收益率热力图(%)")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 新闻情感分析
            st.subheader("📰 市场情感分析")
            news_data = DataManager.get_news_sentiment(symbol)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_sentiment = px.line(news_data, x='Date', y='Sentiment', title='新闻情感趋势')
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                fig_news_count = px.bar(news_data, x='Date', y='News_Count', title='新闻数量统计')
                st.plotly_chart(fig_news_count, use_container_width=True)
            
            # 策略解释
            st.subheader("🤖 AI策略解释")
            explanation = f"""
            **策略逻辑分析:**
            
            1. **选择理由**: {strategy_description}
            2. **参数设置**: {params}
            3. **风险控制**: {'启用止损' + str(stop_loss_pct*100) + '%' if use_stop_loss else '未启用止损'}
            4. **预期表现**: 基于历史数据，该策略的夏普比率为 {metrics['夏普比率']}，最大回撤为 {metrics['最大回撤']}
            
            **市场适应性**: 
            - 该策略在趋势明显的市场中表现较好
            - 建议在横盘震荡市场中降低仓位或暂停使用
            - 需要密切关注市场情绪变化和宏观经济指标
            
            **优化建议**:
            - 可以考虑结合多个时间框架进行确认
            - 建议定期重新优化参数以适应市场变化
            - 可以加入成交量指标作为辅助确认信号
            """
            
            st.markdown(explanation)
            
            # 数据导出功能
            st.subheader("📁 数据导出")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 导出回测数据"):
                    csv = backtest_data.to_csv()
                    st.download_button(
                        label="下载CSV文件",
                        data=csv,
                        file_name=f"{symbol}_{selected_strategy}_backtest.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📈 导出交易信号"):
                    signals = backtest_data[backtest_data['Position'] != 0][['Close', 'Position', 'Signal']].copy()
                    csv_signals = signals.to_csv()
                    st.download_button(
                        label="下载交易信号",
                        data=csv_signals,
                        file_name=f"{symbol}_{selected_strategy}_signals.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("📋 导出策略报告"):
                    report = f"""
# {symbol} {selected_strategy} 策略报告

## 策略参数
{json.dumps(params, indent=2, ensure_ascii=False)}

## 回测指标
{json.dumps(metrics, indent=2, ensure_ascii=False)}

## 策略描述
{strategy_description}

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    st.download_button(
                        label="下载策略报告",
                        data=report,
                        file_name=f"{symbol}_{selected_strategy}_report.md",
                        mime="text/markdown"
                    )

# 增强功能模块
class AdvancedFeatures:
    @staticmethod
    def portfolio_optimization(symbols, weights=None):
        """投资组合优化"""
        if weights is None:
            weights = [1/len(symbols)] * len(symbols)
        
        portfolio_data = {}
        for symbol in symbols:
            data = DataManager.get_yahoo_data(symbol, "1y")
            if data is not None:
                portfolio_data[symbol] = data['Close'].pct_change().dropna()
        
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            
            # 计算协方差矩阵
            cov_matrix = portfolio_df.cov() * 252
            
            # 计算投资组合指标
            portfolio_return = (portfolio_df.mean() * weights).sum() * 252
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_vol
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'weights': dict(zip(symbols, weights))
            }
        return None
    
    @staticmethod
    def monte_carlo_simulation(data, days=252, simulations=1000):
        """蒙特卡洛模拟"""
        returns = data['Close'].pct_change().dropna()
        
        last_price = data['Close'].iloc[-1]
        mean_return = returns.mean()
        std_return = returns.std()
        
        # 模拟价格路径
        simulation_results = []
        
        for _ in range(simulations):
            prices = [last_price]
            for day in range(days):
                random_return = np.random.normal(mean_return, std_return)
                next_price = prices[-1] * (1 + random_return)
                prices.append(next_price)
            simulation_results.append(prices)
        
        return np.array(simulation_results)

# 实时监控模块
class RealTimeMonitor:
    @staticmethod
    def create_alerts(data, strategy_data):
        """创建交易提醒"""
        latest_signal = strategy_data['Signal'].iloc[-1]
        latest_price = data['Close'].iloc[-1]
        
        alerts = []
        
        if latest_signal == 1:
            alerts.append({
                'type': 'BUY',
                'message': f"买入信号触发! 当前价格: ${latest_price:.2f}",
                'timestamp': datetime.now()
            })
        elif latest_signal == -1:
            alerts.append({
                'type': 'SELL',
                'message': f"卖出信号触发! 当前价格: ${latest_price:.2f}",
                'timestamp': datetime.now()
            })
        
        return alerts

# 添加侧边栏的高级功能选项
def add_advanced_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 高级功能")
    
    # 投资组合分析
    if st.sidebar.checkbox("投资组合分析"):
        st.sidebar.text_input("投资组合股票(逗号分隔)", value="AAPL,GOOGL,MSFT", key="portfolio_symbols")
    
    # 蒙特卡洛模拟
    monte_carlo = st.sidebar.checkbox("蒙特卡洛模拟")
    if monte_carlo:
        simulation_days = st.sidebar.slider("模拟天数", 30, 365, 252)
        num_simulations = st.sidebar.slider("模拟次数", 100, 2000, 1000)
    
    # 实时监控
    real_time = st.sidebar.checkbox("实时监控")
    
    return {
        'portfolio_analysis': st.session_state.get('portfolio_symbols'),
        'monte_carlo': monte_carlo,
        'simulation_days': simulation_days if monte_carlo else None,
        'num_simulations': num_simulations if monte_carlo else None,
        'real_time': real_time
    }

# 主函数更新
def main():
    # 标题
    st.markdown('<h1 class="main-header">🚀 QuantGPT Pro - AI量化交易平台</h1>', unsafe_allow_html=True)
    
    # 添加功能选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["📈 策略回测", "📊 投资组合", "🎲 风险模拟", "⚡ 实时监控"])
    
    # 侧边栏
    st.sidebar.title("📊 策略配置")
    
    # 选择数据源
    data_source = st.sidebar.selectbox(
        "选择数据源",
        ["Yahoo Finance", "Alpha Vantage"]
    )
    
    # 股票代码输入
    symbol = st.sidebar.text_input("股票代码", value="AAPL").upper()
    
    # Alpha Vantage API Key（如果选择）
    if data_source == "Alpha Vantage":
        api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
    
    # 时间范围
    period = st.sidebar.selectbox(
        "数据时间范围",
        ["1y", "2y", "5y", "max"]
    )
    
    # 策略选择
    strategy_engine = StrategyEngine()
    selected_strategy = st.sidebar.selectbox(
        "选择策略类型",
        list(strategy_engine.strategies.keys())
    )
    
    # 获取高级功能设置
    advanced_options = add_advanced_sidebar()
    
    with tab1:
        # 原有的策略回测功能保持不变
        # 策略参数设置
        st.sidebar.subheader("策略参数")
        params = {}
        
        if selected_strategy == "趋势跟踪":
            params['short_window'] = st.sidebar.slider("短期窗口", 5, 50, 20)
            params['long_window'] = st.sidebar.slider("长期窗口", 20, 200, 50)
        elif selected_strategy == "均值回归":
            params['window'] = st.sidebar.slider("移动窗口", 10, 50, 20)
            params['std_dev'] = st.sidebar.slider("标准差倍数", 1.0, 3.0, 2.0, 0.1)
        elif selected_strategy == "动量策略":
            params['window'] = st.sidebar.slider("RSI窗口", 5, 30, 14)
            params['threshold'] = st.sidebar.slider("RSI阈值", 60, 80, 70)
        elif selected_strategy == "配对交易":
            params['window'] = st.sidebar.slider("统计窗口", 20, 60, 30)
        elif selected_strategy == "突破策略":
            params['window'] = st.sidebar.slider("突破窗口", 10, 50, 20)
        elif selected_strategy == "网格交易":
            params['grid_size'] = st.sidebar.slider("网格大小(%)", 0.5, 5.0, 2.0, 0.1) / 100
        
        # 风险管理参数
        st.sidebar.subheader("风险管理")
        use_stop_loss = st.sidebar.checkbox("启用止损")
        if use_stop_loss:
            stop_loss_pct = st.sidebar.slider("止损百分比(%)", 1, 10, 5) / 100
        
        risk_per_trade = st.sidebar.slider("单笔风险(%)", 1, 5, 2) / 100
        initial_capital = st.sidebar.number_input("初始资金", value=100000, step=10000)
        
        # 参数优化选项
        st.sidebar.subheader("参数优化")
        enable_optimization = st.sidebar.checkbox("启用参数优化")
        
        if st.sidebar.button("🚀 运行回测"):
            # [原有的回测逻辑保持不变]
            # 获取数据
            with st.spinner("正在获取数据..."):
                if data_source == "Yahoo Finance":
                    data = DataManager.get_yahoo_data(symbol, period)
                else:
                    if 'api_key' in locals() and api_key:
                        data = DataManager.get_alpha_vantage_data(symbol, api_key)
                    else:
                        st.error("请输入Alpha Vantage API Key")
                        return
            
            if data is not None and not data.empty:
                # 添加技术指标
                data = TechnicalIndicators.add_all_indicators(data)
                
                # 参数优化
                if enable_optimization:
                    with st.spinner("正在优化参数..."):
                        if selected_strategy == "趋势跟踪":
                            param_grid = {
                                'short_window': [10, 15, 20, 25],
                                'long_window': [40, 50, 60, 70]
                            }
                        elif selected_strategy == "均值回归":
                            param_grid = {
                                'window': [15, 20, 25, 30],
                                'std_dev': [1.5, 2.0, 2.5]
                            }
                        else:
                            param_grid = [params]  # 默认参数
                        
                        if len(param_grid) > 1:
                            best_params, best_score, _ = ParameterOptimizer.grid_search(
                                data, strategy_engine.strategies[selected_strategy], param_grid
                            )
                            st.success(f"最优参数: {best_params}, 夏普比率: {best_score:.2f}")
                            params = best_params
                
                # 运行策略
                with st.spinner("正在运行策略..."):
                    strategy_data, strategy_description = strategy_engine.strategies[selected_strategy](data.copy(), params)
                    
                    # 添加风险管理
                    if use_stop_loss:
                        strategy_data = RiskManager.add_stop_loss(strategy_data, stop_loss_pct)
                    
                    strategy_data = RiskManager.calculate_position_size(strategy_data, risk_per_trade, initial_capital)
                    
                    # 运行回测
                    backtest_data = BacktestEngine.run_backtest(strategy_data, initial_capital)
                    
                    # 计算指标
                    metrics = BacktestEngine.calculate_metrics(backtest_data)
                
                # 显示策略回测结果（保持原有代码）
                st.subheader("📈 策略表现")
                
                # 策略描述
                st.markdown(f'<div class="strategy-description"><strong>策略说明:</strong> {strategy_description}</div>', 
                           unsafe_allow_html=True)
                
                # 显示回测指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总收益率", metrics["总收益率"])
                with col2:
                    st.metric("夏普比率", metrics["夏普比率"])
                with col3:
                    st.metric("最大回撤", metrics["最大回撤"])
                with col4:
                    st.metric("胜率", metrics["胜率"])
    
    with tab2:
        st.subheader("📊 投资组合分析")
        
        # 投资组合输入
        portfolio_symbols = st.text_input("请输入股票代码（逗号分隔）", value="AAPL,GOOGL,MSFT,TSLA")
        symbols_list = [s.strip().upper() for s in portfolio_symbols.split(",")]
        
        if st.button("分析投资组合"):
            with st.spinner("正在分析投资组合..."):
                # 获取投资组合数据
                portfolio_data = {}
                for symbol in symbols_list:
                    data = DataManager.get_yahoo_data(symbol, "1y")
                    if data is not None:
                        portfolio_data[symbol] = data['Close']
                
                if portfolio_data:
                    portfolio_df = pd.DataFrame(portfolio_data)
                    returns_df = portfolio_df.pct_change().dropna()
                    
                    # 等权重投资组合
                    equal_weights = [1/len(symbols_list)] * len(symbols_list)
                    portfolio_result = AdvancedFeatures.portfolio_optimization(symbols_list, equal_weights)
                    
                    if portfolio_result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("投资组合指标")
                            st.metric("预期年化收益", f"{portfolio_result['expected_return']:.2%}")
                            st.metric("预期年化波动", f"{portfolio_result['volatility']:.2%}")
                            st.metric("夏普比率", f"{portfolio_result['sharpe_ratio']:.2f}")
                        
                        with col2:
                            st.subheader("权重分配")
                            weights_df = pd.DataFrame(list(portfolio_result['weights'].items()), 
                                                    columns=['股票', '权重'])
                            fig_pie = px.pie(weights_df, values='权重', names='股票', title='投资组合权重分配')
                            st.plotly_chart(fig_pie)
                        
                        # 相关性热力图
                        st.subheader("股票相关性分析")
                        corr_matrix = returns_df.corr()
                        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                           title="股票收益率相关性矩阵")
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # 累计收益对比
                        st.subheader("累计收益对比")
                        cumulative_returns = (1 + returns_df).cumprod()
                        fig_cumret = px.line(cumulative_returns, title="各股票累计收益对比")
                        st.plotly_chart(fig_cumret, use_container_width=True)
    
    with tab3:
        st.subheader("🎲 蒙特卡洛风险模拟")
        
        simulation_symbol = st.selectbox("选择模拟股票", ["AAPL", "GOOGL", "MSFT", "TSLA"])
        simulation_days = st.slider("模拟天数", 30, 365, 252)
        num_simulations = st.slider("模拟次数", 100, 2000, 1000)
        
        if st.button("开始蒙特卡洛模拟"):
            with st.spinner("正在运行蒙特卡洛模拟..."):
                data = DataManager.get_yahoo_data(simulation_symbol, "2y")
                if data is not None:
                    simulation_results = AdvancedFeatures.monte_carlo_simulation(
                        data, simulation_days, num_simulations
                    )
                    
                    # 计算统计指标
                    final_prices = simulation_results[:, -1]
                    current_price = data['Close'].iloc[-1]
                    
                    price_change = (final_prices - current_price) / current_price
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("模拟结果统计")
                        st.metric("当前价格", f"${current_price:.2f}")
                        st.metric("预期价格", f"${np.mean(final_prices):.2f}")
                        st.metric("价格中位数", f"${np.median(final_prices):.2f}")
                        st.metric("5%分位数", f"${np.percentile(final_prices, 5):.2f}")
                        st.metric("95%分位数", f"${np.percentile(final_prices, 95):.2f}")
                        
                        # VaR计算
                        var_5 = np.percentile(price_change, 5)
                        st.metric("VaR (5%)", f"{var_5:.2%}")
                    
                    with col2:
                        # 价格分布直方图
                        fig_hist = px.histogram(x=final_prices, nbins=50, 
                                              title=f"{simulation_symbol} {simulation_days}天后价格分布")
                        fig_hist.add_vline(x=current_price, line_dash="dash", 
                                         annotation_text="当前价格")
                        st.plotly_chart(fig_hist)
                    
                    # 模拟路径图
                    st.subheader("价格路径模拟")
                    # 显示前100条路径以避免图表过于拥挤
                    paths_to_show = min(100, num_simulations)
                    fig_paths = go.Figure()
                    
                    for i in range(paths_to_show):
                        fig_paths.add_trace(go.Scatter(
                            y=simulation_results[i],
                            mode='lines',
                            line=dict(width=0.5, color='lightblue'),
                            showlegend=False
                        ))
                    
                    fig_paths.add_trace(go.Scatter(
                        y=np.mean(simulation_results, axis=0),
                        mode='lines',
                        line=dict(width=3, color='red'),
                        name='平均路径'
                    ))
                    
                    fig_paths.update_layout(title=f"{simulation_symbol} 蒙特卡洛价格路径模拟",
                                          xaxis_title="天数", yaxis_title="价格")
                    st.plotly_chart(fig_paths, use_container_width=True)
    
    with tab4:
        st.subheader("⚡ 实时监控")
        
        if st.button("启动实时监控"):
            # 实时数据获取和监控
            monitoring_symbol = st.selectbox("监控股票", ["AAPL", "GOOGL", "MSFT", "TSLA"], key="monitor")
            
            # 创建实时更新的占位符
            price_placeholder = st.empty()
            alert_placeholder = st.empty()
            
            # 模拟实时更新（实际应用中应该使用WebSocket或定时刷新）
            current_data = DataManager.get_yahoo_data(monitoring_symbol, "1d")
            if current_data is not None:
                current_price = current_data['Close'].iloc[-1]
                
                with price_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("当前价格", f"${current_price:.2f}")
                    with col2:
                        daily_change = current_data['Close'].pct_change().iloc[-1]
                        st.metric("日内涨跌", f"{daily_change:.2%}")
                    with col3:
                        volume = current_data['Volume'].iloc[-1] if 'Volume' in current_data.columns else 0
                        st.metric("成交量", f"{volume:,.0f}")
                
                # 检查交易信号
                data_for_strategy = TechnicalIndicators.add_all_indicators(current_data)
                strategy_data, _ = strategy_engine.strategies[selected_strategy](data_for_strategy.copy(), params)
                
                alerts = RealTimeMonitor.create_alerts(current_data, strategy_data)
                
                if alerts:
                    with alert_placeholder.container():
                        for alert in alerts:
                            if alert['type'] == 'BUY':
                                st.success(f"🟢 {alert['message']}")
                            else:
                                st.error(f"🔴 {alert['message']}")

if __name__ == "__main__":
    main()
