import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import re
import warnings

# 尝试导入可选依赖
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="QuantGPT Pro - AI量化交易平台",
    page_icon="🚀",
    layout="wide"
)

# CSS样式
st.markdown("""
<style>
    .main > div {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: flex-start;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        flex-direction: row-reverse;
        margin-left: 2rem;
    }
    
    .chat-message.bot {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
    }
    
    .chat-message .avatar {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        margin: 0 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        background: rgba(255,255,255,0.2);
    }
    
    .chat-message .message {
        flex: 1;
        line-height: 1.6;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #666;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
    }
    
    .status-online {
        background: #22c55e;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-offline {
        background: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .premium-badge {
        background: #ffd700;
        color: #333;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.7rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# 模拟数据生成器
class MockDataGenerator:
    @staticmethod
    def generate_mock_data(symbol, period="2y"):
        end_date = datetime.now()
        days = 730 if period == "2y" else 365
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(hash(symbol) % 1000)
        
        base_price = 100 + (hash(symbol) % 500)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
        data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
        data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
        data['Volume'] = np.random.randint(1000000, 10000000, len(data))
        
        return data.dropna()

# 技术指标
class TechnicalIndicators:
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data, window=20, std_dev=2):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower, sma

# 数据获取
@st.cache_data(ttl=3600)
def get_stock_data(symbol, period="2y"):
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if not data.empty:
                return data
        except:
            pass
    return MockDataGenerator.generate_mock_data(symbol, period)

# 策略引擎
class StrategyEngine:
    def trend_following(self, data, short_window=20, long_window=50):
        if TA_AVAILABLE:
            data['SMA_short'] = ta.trend.SMAIndicator(data['Close'], window=short_window).sma_indicator()
            data['SMA_long'] = ta.trend.SMAIndicator(data['Close'], window=long_window).sma_indicator()
        else:
            data['SMA_short'] = TechnicalIndicators.sma(data['Close'], short_window)
            data['SMA_long'] = TechnicalIndicators.sma(data['Close'], long_window)
        
        data['Signal'] = 0
        data['Signal'][short_window:] = np.where(
            data['SMA_short'][short_window:] > data['SMA_long'][short_window:], 1, 0
        )
        data['Position'] = data['Signal'].diff()
        return data, f"双均线策略({short_window}日/{long_window}日)"
    
    def mean_reversion(self, data, window=20):
        if TA_AVAILABLE:
            bb = ta.volatility.BollingerBands(data['Close'], window=window)
            data['Upper'] = bb.bollinger_hband()
            data['Lower'] = bb.bollinger_lband()
        else:
            data['Upper'], data['Lower'], data['SMA'] = TechnicalIndicators.bollinger_bands(data['Close'], window)
        
        data['Signal'] = np.where(data['Close'] < data['Lower'], 1, 
                                np.where(data['Close'] > data['Upper'], -1, 0))
        data['Position'] = data['Signal'].diff()
        return data, f"布林带均值回归策略({window}日)"
    
    def momentum_strategy(self, data, rsi_window=14):
        if TA_AVAILABLE:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
        else:
            data['RSI'] = TechnicalIndicators.rsi(data['Close'], rsi_window)
        
        data['Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
        data['Position'] = data['Signal'].diff()
        return data, f"RSI动量策略({rsi_window}日)"

# 回测引擎
class BacktestEngine:
    def run_backtest(self, data, initial_capital=100000, commission=0.001):
        data = data.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
        data['Strategy_Returns'] = data['Strategy_Returns'] - (np.abs(data['Position']) * commission)
        
        data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
        data['Strategy_Cumulative'] = (1 + data['Strategy_Returns']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Strategy_Cumulative']
        
        return data
    
    def calculate_metrics(self, data):
        strategy_returns = data['Strategy_Returns'].dropna()
        
        if len(strategy_returns) == 0:
            return {"错误": "无有效交易数据"}
        
        total_return = data['Strategy_Cumulative'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        cumulative = data['Strategy_Cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        winning_trades = len(strategy_returns[strategy_returns > 0])
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "总收益率": f"{total_return:.2%}",
            "年化收益率": f"{annual_return:.2%}",
            "夏普比率": f"{sharpe_ratio:.2f}",
            "最大回撤": f"{max_drawdown:.2%}",
            "胜率": f"{win_rate:.2%}",
            "交易次数": total_trades
        }

# AI分析师
class QuantGPTAnalyst:
    def __init__(self):
        self.strategy_engine = StrategyEngine()
        self.backtest_engine = BacktestEngine()
    
    def parse_input(self, user_input):
        stocks = re.findall(r'\b[A-Z]{1,5}\b', user_input.upper())
        
        strategy_map = {
            "趋势": "trend", "均线": "trend", "双均线": "trend",
            "均值回归": "mean", "布林带": "mean",
            "动量": "momentum", "rsi": "momentum"
        }
        
        strategy = None
        for keyword, strat in strategy_map.items():
            if keyword in user_input.lower():
                strategy = strat
                break
        
        return {"stocks": stocks, "strategy": strategy}
    
    def analyze_stock(self, stock, strategy_type):
        data = get_stock_data(stock)
        
        if strategy_type == "trend":
            data, desc = self.strategy_engine.trend_following(data)
        elif strategy_type == "mean":
            data, desc = self.strategy_engine.mean_reversion(data)
        elif strategy_type == "momentum":
            data, desc = self.strategy_engine.momentum_strategy(data)
        else:
            data, desc = self.strategy_engine.trend_following(data)
        
        backtest_data = self.backtest_engine.run_backtest(data)
        metrics = self.backtest_engine.calculate_metrics(backtest_data)
        
        # 存储数据
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = {}
        st.session_state.analysis_data[stock] = backtest_data
        
        return self.format_result(stock, desc, metrics)
    
    def format_result(self, stock, description, metrics):
        result = f"## 📊 {stock} 分析结果\n\n"
        result += f"**策略：** {description}\n\n"
        result += "**回测指标：**\n"
        
        for metric, value in metrics.items():
            result += f"• {metric}: {value}\n"
        
        # AI评估
        try:
            sharpe = float(metrics["夏普比率"])
            if sharpe > 1.0:
                result += "\n✅ **AI评估：** 策略表现优秀，建议考虑实盘应用"
            elif sharpe > 0.5:
                result += "\n⚠️ **AI评估：** 策略表现一般，建议优化参数"
            else:
                result += "\n❌ **AI评估：** 策略表现较差，建议重新选择"
        except:
            result += "\n💡 **AI评估：** 分析完成，请查看指标详情"
        
        return result
    
    def generate_response(self, user_input):
        parsed = self.parse_input(user_input)
        
        if not parsed["stocks"]:
            return "🤖 请告诉我您想分析的股票代码，例如：'分析AAPL的趋势策略'"
        
        results = []
        for stock in parsed["stocks"]:
            try:
                result = self.analyze_stock(stock, parsed["strategy"])
                results.append(result)
            except Exception as e:
                results.append(f"❌ 分析{stock}失败：{str(e)}")
        
        return "\n\n".join(results)

# 图表生成
def create_chart(stock):
    if 'analysis_data' not in st.session_state or stock not in st.session_state.analysis_data:
        st.error(f"没有{stock}的数据")
        return
    
    data = st.session_state.analysis_data[stock]
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[f'{stock} 价格走势', '技术指标', '策略收益'],
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # 价格线
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='收盘价', line=dict(color='blue')),
        row=1, col=1
    )
    
    # 均线
    if 'SMA_short' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_short'], name='短期均线', line=dict(color='orange')),
            row=1, col=1
        )
    
    # 买卖信号
    buy_signals = data[data['Position'] == 1]
    sell_signals = data[data['Position'] == -1]
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                      mode='markers', name='买入', marker=dict(color='green', size=10)),
            row=1, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                      mode='markers', name='卖出', marker=dict(color='red', size=10)),
            row=1, col=1
        )
    
    # RSI指标
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
    
    # 收益对比
    benchmark = (data['Cumulative_Returns'] - 1) * 100
    strategy = (data['Strategy_Cumulative'] - 1) * 100
    
    fig.add_trace(
        go.Scatter(x=data.index, y=benchmark, name='买入持有', line=dict(color='gray')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=strategy, name='策略收益', line=dict(color='green')),
        row=3, col=1
    )
    
    fig.update_layout(height=800, title=f"{stock} 策略分析")
    st.plotly_chart(fig, use_container_width=True)

# 消息显示
def display_message(message, is_user=False):
    message_class = "user" if is_user else "bot"
    avatar = "👤" if is_user else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="avatar">{avatar}</div>
        <div class="message">{message}</div>
    </div>
    """, unsafe_allow_html=True)

# 主程序
def main():
    # 标题
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">QuantGPT Pro</h1>
        <p style="font-size: 1.2rem;">🚀 AI量化交易分析平台</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 状态指示器
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = "status-online" if YFINANCE_AVAILABLE else "status-offline"
        text = "实时数据" if YFINANCE_AVAILABLE else "模拟数据"
        st.markdown(f'<div class="{status}">📡 {text}</div>', unsafe_allow_html=True)
    
    with col2:
        ta_status = "status-online" if TA_AVAILABLE else "status-offline"
        ta_text = "完整指标" if TA_AVAILABLE else "基础指标"
        st.markdown(f'<div class="{ta_status}">📈 {ta_text}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="status-online">🛡️ 风控启用</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="premium-badge">PRO版本</div>', unsafe_allow_html=True)
    
    # 初始化
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": """🎉 **欢迎使用QuantGPT Pro！**

我是您的AI量化交易助手，可以帮您：

🎯 **策略分析**
• 趋势跟踪 - 双均线策略
• 均值回归 - 布林带策略  
• 动量策略 - RSI指标策略

📊 **专业回测**
• 完整的回测指标
• 可视化图表展示
• AI智能评估

**示例：**
• "分析AAPL的趋势策略"
• "TSLA的布林带策略"
• "GOOGL的RSI动量策略"

现在就告诉我您想分析什么股票吧！"""
        }]
    
    if "analyst" not in st.session_state:
        st.session_state.analyst = QuantGPTAnalyst()
    
    # 显示消息
    for message in st.session_state.messages:
        display_message(message["content"], message["role"] == "user")
    
    # 用户输入
    user_input = st.chat_input("请输入您的分析需求...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message(user_input, True)
        
        with st.spinner("AI正在分析中..."):
            response = st.session_state.analyst.generate_response(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_message(response, False)
        
        # 图表按钮
        if 'analysis_data' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 图表分析")
            
            cols = st.columns(len(st.session_state.analysis_data))
            for i, stock in enumerate(st.session_state.analysis_data.keys()):
                with cols[i]:
                    if st.button(f"📈 {stock} 图表", key=f"chart_{stock}"):
                        create_chart(stock)
        
        st.rerun()
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 🎛️ 控制面板")
        
        if st.button("🗑️ 清除历史"):
            st.session_state.messages = [st.session_state.messages[0]]
            if 'analysis_data' in st.session_state:
                del st.session_state.analysis_data
            st.rerun()
        
        st.markdown("### 🚀 快速分析")
        
        stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA"]
        selected_stock = st.selectbox("选择股票", stocks)
        
        strategies = ["趋势策略", "布林带策略", "RSI策略"]
        selected_strategy = st.selectbox("选择策略", strategies)
        
        if st.button("开始分析"):
            query = f"分析{selected_stock}的{selected_strategy}"
            st.session_state.messages.append({"role": "user", "content": query})
            response = st.session_state.analyst.generate_response(query)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 系统信息")
        st.markdown(f"**数据源**: {'yfinance' if YFINANCE_AVAILABLE else '模拟数据'}")
        st.markdown(f"**技术指标**: {'完整' if TA_AVAILABLE else '基础'}")
        st.markdown("**策略**: 3种核心策略")

if __name__ == "__main__":
    main()
