import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import re
import time
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
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
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .chat-container {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: flex-start;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .chat-message:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
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
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .chat-message .message {
        flex: 1;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #666;
        font-weight: 500;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-online {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        color: white;
    }
    
    .status-limited {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .premium-badge {
        background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .quick-action-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .quick-action-card {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        backdrop-filter: blur(10px);
    }
    
    .quick-action-card:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: rgba(255,255,255,0.7);
        margin: 0 2px;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0.2s; }
    .typing-dot:nth-child(2) { animation-delay: 0.4s; }
    .typing-dot:nth-child(3) { animation-delay: 0.6s; }
    
    @keyframes typing {
        0%, 80%, 100% { opacity: 0.3; }
        40% { opacity: 1; }
    }
    
    .chart-container {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .sidebar .block-container {
        padding: 1.5rem;
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# 数据缓存装饰器
@st.cache_data(ttl=3600)
def get_cached_stock_data(symbol, period="2y"):
    """缓存股票数据"""
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if not data.empty:
                return data
        except Exception as e:
            st.warning(f"获取{symbol}数据失败: {str(e)}")
    
    # 返回模拟数据
    return MockDataGenerator.generate_mock_data(symbol, period)

# 模拟数据生成器
class MockDataGenerator:
    @staticmethod
    def generate_mock_data(symbol, period="2y"):
        """生成高质量模拟数据"""
        end_date = datetime.now()
        days_map = {"2y": 730, "1y": 365, "6mo": 180, "3mo": 90, "1mo": 30}
        days = days_map.get(period, 730)
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(hash(symbol) % 1000)
        
        # 生成更真实的价格走势
        base_price = 100 + (hash(symbol) % 500)
        trend = np.random.choice([-0.0002, 0.0002, 0.0005], p=[0.3, 0.4, 0.3])
        volatility = 0.015 + (hash(symbol) % 100) / 10000
        
        prices = [base_price]
        for i in range(1, len(dates)):
            # 加入趋势和季节性
            seasonal = 0.001 * np.sin(2 * np.pi * i / 252)
            noise = np.random.normal(trend + seasonal, volatility)
            new_price = prices[-1] * (1 + noise)
            prices.append(max(new_price, prices[-1] * 0.9))
        
        # 创建OHLC数据
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.002, len(data)))
        data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.008, len(data))))
        data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.008, len(data))))
        data['Volume'] = np.random.lognormal(15, 0.5, len(data)).astype(int)
        
        return data.dropna()

# 技术指标计算
class TechnicalIndicators:
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        return data.ewm(span=window, adjust=False).mean()
    
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
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

# 策略引擎
class StrategyEngine:
    def __init__(self):
        self.strategies = {
            "趋势跟踪": self.trend_following,
            "均值回归": self.mean_reversion,
            "动量策略": self.momentum_strategy,
            "突破策略": self.breakout_strategy,
            "网格交易": self.grid_trading,
            "配对交易": self.pairs_trading
        }
    
    def trend_following(self, data, params):
        """趋势跟踪策略"""
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
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
        
        return data, f"使用{short_window}日和{long_window}日双均线策略"
    
    def mean_reversion(self, data, params):
        """均值回归策略"""
        window = params.get('window', 20)
        std_dev = params.get('std_dev', 2.0)
        
        if TA_AVAILABLE:
            bb = ta.volatility.BollingerBands(data['Close'], window=window, window_dev=std_dev)
            data['Upper'] = bb.bollinger_hband()
            data['Lower'] = bb.bollinger_lband()
            data['SMA'] = bb.bollinger_mavg()
        else:
            data['Upper'], data['Lower'], data['SMA'] = TechnicalIndicators.bollinger_bands(
                data['Close'], window, std_dev
            )
        
        data['Signal'] = 0
        data['Signal'] = np.where(data['Close'] < data['Lower'], 1, 
                                np.where(data['Close'] > data['Upper'], -1, 0))
        data['Position'] = data['Signal'].diff()
        
        return data, f"布林带均值回归策略，窗口{window}日"
    
    def momentum_strategy(self, data, params):
        """动量策略"""
        rsi_window = params.get('rsi_window', 14)
        
        if TA_AVAILABLE:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
            data['MACD'] = ta.trend.MACD(data['Close']).macd()
            data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
        else:
            data['RSI'] = TechnicalIndicators.rsi(data['Close'], rsi_window)
            data['MACD'], data['MACD_signal'], _ = TechnicalIndicators.macd(data['Close'])
        
        data['Signal'] = 0
        data['Signal'] = np.where(
            (data['RSI'] > 70) & (data['MACD'] > data['MACD_signal']), 1,
            np.where((data['RSI'] < 30) & (data['MACD'] < data['MACD_signal']), -1, 0)
        )
        data['Position'] = data['Signal'].diff()
        
        return data, f"RSI和MACD组合动量策略"
    
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
        
        return data, f"唐奇安通道突破策略，窗口{window}日"
    
    def grid_trading(self, data, params):
        """网格交易策略"""
        grid_size = params.get('grid_size', 0.02)
        
        data['Price_change'] = data['Close'].pct_change().cumsum()
        data['Grid_level'] = (data['Price_change'] / grid_size).round()
        data['Signal'] = -data['Grid_level'].diff()
        data['Position'] = data['Signal'].diff()
        
        return data, f"网格交易策略，间距{grid_size*100:.1f}%"
    
    def pairs_trading(self, data, params):
        """配对交易策略"""
        window = params.get('window', 30)
        
        data['MA'] = data['Close'].rolling(window=window).mean()
        data['Spread'] = data['Close'] - data['MA']
        data['Spread_MA'] = data['Spread'].rolling(window=window).mean()
        data['Spread_STD'] = data['Spread'].rolling(window=window).std()
        
        data['Signal'] = np.where(
            data['Spread'] > data['Spread_MA'] + data['Spread_STD'], -1,
            np.where(data['Spread'] < data['Spread_MA'] - data['Spread_STD'], 1, 0)
        )
        data['Position'] = data['Signal'].diff()
        
        return data, f"统计套利策略，{window}日均值回归"

# 回测引擎
class BacktestEngine:
    @staticmethod
    def run_backtest(data, initial_capital=100000, commission=0.001):
        """运行回测"""
        data = data.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
        data['Strategy_Returns'] = data['Strategy_Returns'] - (np.abs(data['Position']) * commission)
        
        data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
        data['Strategy_Cumulative'] = (1 + data['Strategy_Returns']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Strategy_Cumulative']
        
        return data
    
    @staticmethod
    def calculate_metrics(data):
        """计算回测指标"""
        strategy_returns = data['Strategy_Returns'].dropna()
        
        if len(strategy_returns) == 0:
            return {"错误": "无有效交易数据"}
        
        total_return = data['Strategy_Cumulative'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1 if len(strategy_returns) > 0 else 0
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
            "交易次数": total_trades,
            "年化波动率": f"{volatility:.2%}"
        }

# AI分析师
class QuantGPTAnalyst:
    def __init__(self):
        self.strategy_engine = StrategyEngine()
        self.backtest_engine = BacktestEngine()
        
        self.strategy_library = {
            "保守型": {"strategies": ["均值回归", "网格交易"]},
            "平衡型": {"strategies": ["趋势跟踪", "突破策略"]},
            "激进型": {"strategies": ["动量策略"]},
            "专业型": {"strategies": ["配对交易"]}
        }
    
    def parse_user_input(self, user_input):
        """解析用户输入"""
        user_input_lower = user_input.lower()
        
        # 提取股票代码
        stock_pattern = r'\b[A-Z]{1,5}\b'
        stocks = re.findall(stock_pattern, user_input.upper())
        
        # 提取策略类型
        strategy_keywords = {
            "趋势": "趋势跟踪", "均线": "趋势跟踪", "双均线": "趋势跟踪",
            "均值回归": "均值回归", "布林带": "均值回归",
            "动量": "动量策略", "rsi": "动量策略", "macd": "动量策略",
            "突破": "突破策略", "通道": "突破策略",
            "网格": "网格交易", "配对": "配对交易", "套利": "配对交易"
        }
        
        detected_strategy = None
        for keyword, strategy in strategy_keywords.items():
            if keyword in user_input_lower:
                detected_strategy = strategy
                break
        
        # 风险偏好识别
        risk_keywords = {
            "保守": "保守型", "稳健": "保守型",
            "平衡": "平衡型", "中等": "平衡型",
            "激进": "激进型", "专业": "专业型"
        }
        
        risk_preference = None
        for keyword, risk_type in risk_keywords.items():
            if keyword in user_input_lower:
                risk_preference = risk_type
                break
        
        # 参数提取
        params = {}
        numbers = re.findall(r'\d+', user_input)
        if any(word in user_input_lower for word in ["天", "日"]):
            if len(numbers) >= 1:
                params['window'] = int(numbers[0])
                params['short_window'] = int(numbers[0])
            if len(numbers) >= 2:
                params['long_window'] = int(numbers[1])
        
        percentage_matches = re.findall(r'(\d+(?:\.\d+)?)%', user_input)
        if percentage_matches:
            if "止损" in user_input_lower:
                params['stop_loss'] = float(percentage_matches[0]) / 100
            elif "网格" in user_input_lower:
                params['grid_size'] = float(percentage_matches[0]) / 100
        
        period_keywords = {
            "1月": "1mo", "3月": "3mo", "半年": "6mo", "1年": "1y", "2年": "2y"
        }
        
        period = "2y"
        for keyword, p in period_keywords.items():
            if keyword in user_input:
                period = p
                break
        
        return {
            'stocks': list(set(stocks)),
            'strategy': detected_strategy,
            'risk_preference': risk_preference,
            'params': params,
            'period': period,
            'original_input': user_input
        }
    
    def generate_response(self, parsed_input):
        """生成AI响应"""
        stocks = parsed_input['stocks']
        strategy = parsed_input['strategy']
        risk_preference = parsed_input['risk_preference']
        params = parsed_input['params']
        period = parsed_input['period']
        
        if not stocks:
            return """🤖 **QuantGPT Pro 为您服务！**

请告诉我您想分析的股票代码，例如：
• "分析AAPL的趋势策略"
• "用保守型策略分析TSLA"
• "GOOGL的动量策略分析"
"""
        
        if not strategy and not risk_preference:
            return f"""🤖 **检测到股票：{', '.join(stocks)}**

请选择分析策略或风险偏好：

🎯 **策略选择：**
• **趋势跟踪** - 双均线系统
• **均值回归** - 布林带策略
• **动量策略** - RSI+MACD组合
• **突破策略** - 通道突破
• **网格交易** - 区间操作
• **配对交易** - 统计套利

🎨 **风险偏好：**
• **保守型** - 稳健策略
• **平衡型** - 均衡配置
• **激进型** - 高收益策略
• **专业型** - 定制策略

示例："用平衡型策略分析{stocks[0]}"
"""
        
        # 如果有风险偏好但没有具体策略，推荐策略
        if risk_preference and not strategy:
            return self._generate_risk_based_response(stocks, risk_preference)
        
        # 执行分析
        return self._execute_analysis(stocks, strategy, params, period)
    
    def _generate_risk_based_response(self, stocks, risk_preference):
        """基于风险偏好生成响应"""
        strategy_info = self.strategy_library[risk_preference]
        recommended_strategies = strategy_info["strategies"]
        
        response = f"🤖 **{risk_preference}投资者 - {', '.join(stocks)}分析**\n\n"
        response += f"基于您的**{risk_preference}**偏好，推荐策略：\n\n"
        
        results = []
        for strategy in recommended_strategies:
            try:
                result = self._analyze_single_strategy(stocks[0], strategy, {}, "1y")
                results.append(result)
            except Exception as e:
                results.append(f"❌ {strategy}分析失败：{str(e)}")
        
        return response + "\n\n".join(results)
    
    def _execute_analysis(self, stocks, strategy, params, period):
        """执行完整分析"""
        results = []
        
        for stock in stocks:
            try:
                result = self._analyze_single_strategy(stock, strategy, params, period)
                results.append(result)
            except Exception as e:
                results.append(f"
