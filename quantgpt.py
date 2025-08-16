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
    page_title="QuantGPT Pro - AI Quantitative Trading Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# OpenQuant风格CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 全局重置 */
    .main > div {
        padding: 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: #FAFBFC;
    }
    
    /* 隐藏Streamlit默认元素 */
    .stApp > header {
        background: transparent;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 顶部导航栏 */
    .top-nav {
        background: white;
        padding: 1rem 2rem;
        border-bottom: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .logo {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1F2937;
        text-decoration: none;
    }
    
    .logo-subtitle {
        font-size: 0.875rem;
        color: #6B7280;
        font-weight: 500;
    }
    
    .nav-status {
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-online {
        background: #DCFCE7;
        color: #15803D;
    }
    
    .status-offline {
        background: #FEE2E2;
        color: #DC2626;
    }
    
    .status-warning {
        background: #FEF3C7;
        color: #D97706;
    }
    
    /* 主要内容区域 */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* 英雄区域 */
    .hero-section {
        background: white;
        border-radius: 12px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid #E5E7EB;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.125rem;
        color: #6B7280;
        margin-bottom: 2rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .feature-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .feature-card:hover {
        border-color: #CBD5E1;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.875rem;
        color: #6B7280;
        line-height: 1.5;
    }
    
    /* 聊天界面 */
    .chat-container {
        background: white;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        overflow: hidden;
    }
    
    .chat-header {
        background: #F9FAFB;
        border-bottom: 1px solid #E5E7EB;
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: #1F2937;
        font-size: 0.875rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .chat-messages {
        padding: 1.5rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .message {
        margin-bottom: 1.5rem;
        max-width: 100%;
    }
    
    .message.user {
        display: flex;
        justify-content: flex-end;
    }
    
    .message.assistant {
        display: flex;
        justify-content: flex-start;
    }
    
    .message-content {
        max-width: 80%;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .message.user .message-content {
        background: #3B82F6;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .message.assistant .message-content {
        background: #F3F4F6;
        color: #1F2937;
        border-bottom-left-radius: 4px;
        border: 1px solid #E5E7EB;
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.875rem;
        margin: 0 12px;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: #3B82F6;
        color: white;
    }
    
    .assistant-avatar {
        background: #10B981;
        color: white;
    }
    
    /* 输入框 */
    .stChatInput > div > div > div > div {
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        background: white !important;
    }
    
    .stChatInput > div > div > div > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: #3B82F6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .secondary-button {
        background: white !important;
        color: #374151 !important;
        border: 1px solid #D1D5DB !important;
    }
    
    .secondary-button:hover {
        background: #F9FAFB !important;
        border-color: #9CA3AF !important;
    }
    
    /* 侧边栏 */
    .sidebar .block-container {
        background: white;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
    }
    
    .sidebar-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* 选择框 */
    .stSelectbox > div > div {
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        background: white !important;
    }
    
    /* 指标卡片 */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #CBD5E1;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-positive {
        color: #10B981;
    }
    
    .metric-negative {
        color: #EF4444;
    }
    
    /* 图表容器 */
    .chart-container {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .chart-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 1rem;
    }
    
    /* 结果卡片 */
    .result-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .result-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #E5E7EB;
    }
    
    .result-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1F2937;
    }
    
    .result-badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-success {
        background: #DCFCE7;
        color: #15803D;
    }
    
    .badge-warning {
        background: #FEF3C7;
        color: #D97706;
    }
    
    .badge-error {
        background: #FEE2E2;
        color: #DC2626;
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main-container {
            padding: 1rem;
        }
        
        .top-nav {
            padding: 1rem;
        }
        
        .hero-section {
            padding: 2rem 1rem;
        }
        
        .hero-title {
            font-size: 2rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .message-content {
            max-width: 90%;
        }
    }
    
    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

# 多语言支持
LANGUAGES = {
    "中文": {
        "title": "QuantGPT Pro",
        "subtitle": "专业AI量化交易分析平台",
        "hero_title": "智能量化交易分析",
        "hero_desc": "使用AI驱动的专业量化分析工具，获得数据驱动的投资洞察",
        "chat_title": "AI量化分析师",
        "data_source": "数据源",
        "technical_indicators": "技术指标",
        "system_status": "系统状态",
        "real_time": "实时数据",
        "simulated": "模拟数据",
        "complete": "完整版",
        "basic": "基础版",
        "online": "在线",
        "quick_analysis": "快速分析",
        "select_stock": "选择股票",
        "select_strategy": "选择策略",
        "start_analysis": "开始分析",
        "clear_history": "清除历史",
        "trend_strategy": "趋势策略",
        "mean_reversion": "均值回归",
        "momentum_strategy": "动量策略",
        "examples": "示例查询",
        "features": {
            "intelligent": {
                "title": "智能分析",
                "desc": "AI驱动的量化策略分析"
            },
            "professional": {
                "title": "专业回测",
                "desc": "完整的回测指标和风险评估"
            },
            "realtime": {
                "title": "实时数据",
                "desc": "获取最新的市场数据"
            },
            "multilingual": {
                "title": "中英双语",
                "desc": "支持中文和英文交互"
            }
        },
        "welcome": """👋 **欢迎使用QuantGPT Pro！**

我是您的专业AI量化分析师，可以帮助您：

**🎯 核心功能**
• **趋势策略** - 双均线交易系统分析
• **均值回归** - 布林带策略回测
• **动量策略** - RSI技术指标分析

**📊 专业服务**
• 完整的量化回测分析
• 风险调整收益评估
• 实时交易信号生成

**💬 使用示例**
• "分析苹果公司的趋势策略"
• "分析AAPL的trend strategy"
• "特斯拉的布林带策略分析"

现在开始您的量化分析之旅吧！"""
    },
    "English": {
        "title": "QuantGPT Pro",
        "subtitle": "Professional AI Quantitative Trading Platform",
        "hero_title": "Intelligent Quantitative Analysis",
        "hero_desc": "Get data-driven investment insights with AI-powered professional quantitative analysis tools",
        "chat_title": "AI Quantitative Analyst",
        "data_source": "Data Source",
        "technical_indicators": "Technical Indicators",
        "system_status": "System Status",
        "real_time": "Real-time Data",
        "simulated": "Simulated Data",
        "complete": "Complete",
        "basic": "Basic",
        "online": "Online",
        "quick_analysis": "Quick Analysis",
        "select_stock": "Select Stock",
        "select_strategy": "Select Strategy",
        "start_analysis": "Start Analysis",
        "clear_history": "Clear History",
        "trend_strategy": "Trend Strategy",
        "mean_reversion": "Mean Reversion",
        "momentum_strategy": "Momentum Strategy",
        "examples": "Example Queries",
        "features": {
            "intelligent": {
                "title": "Intelligent Analysis",
                "desc": "AI-driven quantitative strategy analysis"
            },
            "professional": {
                "title": "Professional Backtesting",
                "desc": "Complete backtesting metrics and risk assessment"
            },
            "realtime": {
                "title": "Real-time Data",
                "desc": "Access to latest market data"
            },
            "multilingual": {
                "title": "Bilingual Support",
                "desc": "Chinese and English interaction"
            }
        },
        "welcome": """👋 **Welcome to QuantGPT Pro!**

I'm your professional AI quantitative analyst, here to help you with:

**🎯 Core Features**
• **Trend Strategy** - Moving average trading system analysis
• **Mean Reversion** - Bollinger Bands strategy backtesting
• **Momentum Strategy** - RSI technical indicator analysis

**📊 Professional Services**
• Complete quantitative backtesting analysis
• Risk-adjusted return evaluation
• Real-time trading signal generation

**💬 Usage Examples**
• "Analyze AAPL trend strategy"
• "分析苹果公司的趋势策略"
• "Tesla Bollinger Bands strategy analysis"

Start your quantitative analysis journey now!"""
    }
}

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
    def __init__(self, lang="中文"):
        self.lang = lang
    
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
        
        if self.lang == "中文":
            desc = f"双均线趋势策略 ({short_window}日/{long_window}日)"
        else:
            desc = f"Moving Average Trend Strategy ({short_window}d/{long_window}d)"
        
        return data, desc
    
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
        
        if self.lang == "中文":
            desc = f"布林带均值回归策略 ({window}日)"
        else:
            desc = f"Bollinger Bands Mean Reversion ({window}d)"
        
        return data, desc
    
    def momentum_strategy(self, data, rsi_window=14):
        if TA_AVAILABLE:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
        else:
            data['RSI'] = TechnicalIndicators.rsi(data['Close'], rsi_window)
        
        data['Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
        data['Position'] = data['Signal'].diff()
        
        if self.lang == "中文":
            desc = f"RSI动量策略 ({rsi_window}日)"
        else:
            desc = f"RSI Momentum Strategy ({rsi_window}d)"
        
        return data, desc

# 回测引擎
class BacktestEngine:
    def __init__(self, lang="中文"):
        self.lang = lang
    
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
            return {"Error": "No valid data" if self.lang == "English" else "无有效数据"}
        
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
        
        if self.lang == "中文":
            return {
                "总收益率": f"{total_return:.2%}",
                "年化收益率": f"{annual_return:.2%}",
                "夏普比率": f"{sharpe_ratio:.2f}",
                "最大回撤": f"{max_drawdown:.2%}",
                "胜率": f"{win_rate:.2%}",
                "交易次数": total_trades
            }
        else:
            return {
                "Total Return": f"{total_return:.2%}",
                "Annual Return": f"{annual_return:.2%}",
                "Sharpe Ratio": f"{sharpe_ratio:.2f}",
                "Max Drawdown": f"{max_drawdown:.2%}",
                "Win Rate": f"{win_rate:.2%}",
                "Trade Count": total_trades
            }

# AI分析师
class BilingualQuantGPTAnalyst:
    def __init__(self, lang="中文"):
        self.lang = lang
        self.strategy_engine = StrategyEngine(lang)
        self.backtest_engine = BacktestEngine(lang)
    
    def detect_language(self, text):
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        return "中文" if chinese_chars > english_chars else "English"
    
    def parse_input(self, user_input):
        detected_lang = self.detect_language(user_input)
        
        stocks = re.findall(r'\b[A-Z]{1,5}\b', user_input.upper())
        
        chinese_stocks = {
            "苹果": "AAPL", "苹果公司": "AAPL", "apple": "AAPL",
            "特斯拉": "TSLA", "tesla": "TSLA",
            "谷歌": "GOOGL", "google": "GOOGL",
            "微软": "MSFT", "microsoft": "MSFT",
            "英伟达": "NVDA", "nvidia": "NVDA",
            "亚马逊": "AMZN", "amazon": "AMZN"
        }
        
        for chinese_name, symbol in chinese_stocks.items():
            if chinese_name in user_input.lower():
                stocks.append(symbol)
        
        strategy_keywords = {
            "趋势": "trend", "均线": "trend", "双均线": "trend", "moving average": "trend", "trend": "trend",
            "均值回归": "mean", "布林带": "mean", "bollinger": "mean", "mean reversion": "mean",
            "动量": "momentum", "rsi": "momentum", "momentum": "momentum"
        }
        
        strategy = None
        for keyword, strat in strategy_keywords.items():
            if keyword in user_input.lower():
                strategy = strat
                break
        
        return {
            "stocks": list(set(stocks)), 
            "strategy": strategy,
            "language": detected_lang
        }
    
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
        
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = {}
        st.session_state.analysis_data[stock] = backtest_data
        
        return self.format_result(stock, desc, metrics)
    
    def format_result(self, stock, description, metrics):
        if self.lang == "中文":
            result = f"## 📊 {stock} 量化分析报告\n\n"
            result += f"**策略描述：** {description}\n\n"
            result += "**核心指标：**\n"
        else:
            result = f"## 📊 {stock} Quantitative Analysis Report\n\n"
            result += f"**Strategy:** {description}\n\n"
            result += "**Key Metrics:**\n"
        
        for metric, value in metrics.items():
            result += f"• **{metric}**: {value}\n"
        
        try:
            sharpe_key = "夏普比率" if self.lang == "中文" else "Sharpe Ratio"
            sharpe = float(metrics[sharpe_key])
            
            result += "\n### 🎯 AI评估\n\n" if self.lang == "中文" else "\n### 🎯 AI Assessment\n\n"
            
            if sharpe > 1.0:
                if self.lang == "中文":
                    result += "✅ **优秀策略** - 夏普比率 > 1.0，风险调整收益表现出色\n"
                    result += "💡 **建议** - 可考虑实盘测试，建议5-10%仓位"
                else:
                    result += "✅ **Excellent Strategy** - Sharpe ratio > 1.0, outstanding risk-adjusted returns\n"
                    result += "💡 **Recommendation** - Consider live testing with 5-10% position size"
            elif sharpe > 0.5:
                if self.lang == "中文":
                    result += "⚠️ **中等策略** - 有一定价值，建议优化参数\n"
                    result += "💡 **建议** - 可小仓位测试或结合其他策略"
                else:
                    result += "⚠️ **Moderate Strategy** - Has value, recommend parameter optimization\n"
                    result += "💡 **Recommendation** - Test with small position or combine with other strategies"
            else:
                if self.lang == "中文":
                    result += "❌ **需要改进** - 表现不佳，建议重新评估\n"
                    result += "💡 **建议** - 尝试其他策略或调整参数"
                else:
                    result += "❌ **Needs Improvement** - Poor performance, recommend reassessment\n"
                    result += "💡 **Recommendation** - Try other strategies or adjust parameters"
        except:
            result += "\n💡 分析完成" if self.lang == "中文" else "\n💡 Analysis completed"
        
        return result
    
    def generate_response(self, user_input):
        parsed = self.parse_input(user_input)
        
        if parsed["language"] != self.lang:
            self.lang = parsed["language"]
            self.strategy_engine.lang = parsed["language"]
            self.backtest_engine.lang = parsed["language"]
        
        if not parsed["stocks"]:
            if self.lang == "中文":
                return "🤖 请指定要分析的股票，如：'分析AAPL的趋势策略' 或 '苹果公司的布林带策略'"
            else:
                return "🤖 Please specify a stock to analyze, e.g., 'Analyze AAPL trend strategy' or 'Apple Bollinger Bands strategy'"
        
        results = []
        for stock in parsed["stocks"]:
            try:
                result = self.analyze_stock(stock, parsed["strategy"])
                results.append(result)
            except Exception as e:
                error_msg = f"❌ 分析{stock}失败：{str(e)}" if self.lang == "中文" else f"❌ Failed to analyze {stock}: {str(e)}"
                results.append(error_msg)
        
        return "\n\n".join(results)

# 图表生成
def create_openquant_style_chart(stock):
    if 'analysis_data' not in st.session_state or stock not in st.session_state.analysis_data:
        st.error(f"No data available for {stock}")
        return
    
    data = st.session_state.analysis_data[stock]
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[f'{stock} Price & Signals', 'Technical Indicators', 'Strategy Performance'],
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.08
    )
    
    # 价格线 - OpenQuant风格
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['Close'], 
            name='Price',
            line=dict(color='#3B82F6', width=2.5),
            hovertemplate='<b>Price</b>: $%{y:.2f}<br><b>Date</b>: %{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 均线
    if 'SMA_short' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['SMA_short'], 
                name='Short MA',
                line=dict(color='#10B981', width=2, dash='dot'),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    if 'SMA_long' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['SMA_long'], 
                name='Long MA',
                line=dict(color='#F59E0B', width=2, dash='dot'),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # 交易信号
    buy_signals = data[data['Position'] == 1]
    sell_signals = data[data['Position'] == -1]
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index, 
                y=buy_signals['Close'],
                mode='markers', 
                name='Buy',
                marker=dict(color='#10B981', size=10, symbol='triangle-up'),
                hovertemplate='<b>Buy Signal</b><br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index, 
                y=sell_signals['Close'],
                mode='markers', 
                name='Sell',
                marker=dict(color='#EF4444', size=10, symbol='triangle-down'),
                hovertemplate='<b>Sell Signal</b><br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # RSI指标
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['RSI'], 
                name='RSI',
                line=dict(color='#8B5CF6', width=2),
                hovertemplate='<b>RSI</b>: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="#EF4444", opacity=0.6)
        fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="#10B981", opacity=0.6)
    
    # 收益对比
    benchmark = (data['Cumulative_Returns'] - 1) * 100
    strategy = (data['Strategy_Cumulative'] - 1) * 100
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=benchmark, 
            name='Buy & Hold',
            line=dict(color='#9CA3AF', width=2),
            hovertemplate='<b>Buy & Hold</b>: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=strategy, 
            name='Strategy',
            line=dict(color='#3B82F6', width=3),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.1)',
            hovertemplate='<b>Strategy</b>: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # OpenQuant风格布局
    fig.update_layout(
        height=700,
        title={
            'text': f"<b>{stock}</b> Quantitative Analysis",
            'x': 0.5,
            'font': {'size': 18, 'color': '#1F2937', 'family': 'Inter'}
        },
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#374151', family='Inter'),
        legend=dict(
            bgcolor='white',
            bordercolor='#E5E7EB',
            borderwidth=1,
            font=dict(color='#374151', size=12)
        ),
        hovermode='x unified'
    )
    
    # 更新坐标轴
    for i in range(1, 4):
        fig.update_xaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='#F3F4F6',
            showline=True,
            linecolor='#E5E7EB',
            row=i, col=1
        )
        fig.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='#F3F4F6',
            showline=True,
            linecolor='#E5E7EB',
            row=i, col=1
        )
    
    return fig

# 消息显示
def display_message(message, is_user=False, lang="中文"):
    if is_user:
        st.markdown(f"""
        <div class="message user">
            <div class="message-avatar user-avatar">U</div>
            <div class="message-content">{message}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message assistant">
            <div class="message-avatar assistant-avatar">AI</div>
            <div class="message-content">{message}</div>
        </div>
        """, unsafe_allow_html=True)

# 主程序
def main():
    # 语言设置
    if 'language' not in st.session_state:
        st.session_state.language = "中文"
    
    lang = st.session_state.language
    t = LANGUAGES[lang]
    
    # 顶部导航
    status_data = "status-online" if YFINANCE_AVAILABLE else "status-offline"
    data_text = t["real_time"] if YFINANCE_AVAILABLE else t["simulated"]
    
    status_ta = "status-online" if TA_AVAILABLE else "status-warning"
    ta_text = t["complete"] if TA_AVAILABLE else t["basic"]
    
    st.markdown(f"""
    <div class="top-nav">
        <div class="logo-section">
            <div class="logo">📊 {t['title']}</div>
            <div class="logo-subtitle">{t['subtitle']}</div>
        </div>
        <div class="nav-status">
            <div class="status-badge {status_data}">{t['data_source']}: {data_text}</div>
            <div class="status-badge {status_ta}">{t['technical_indicators']}: {ta_text}</div>
            <div class="status-badge status-online">{t['system_status']}: {t['online']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 主容器
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # 英雄区域
    st.markdown(f"""
    <div class="hero-section">
        <h1 class="hero-title">{t['hero_title']}</h1>
        <p class="hero-subtitle">{t['hero_desc']}</p>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">{t['features']['intelligent']['title']}</div>
                <div class="feature-desc">{t['features']['intelligent']['desc']}</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">{t['features']['professional']['title']}</div>
                <div class="feature-desc">{t['features']['professional']['desc']}</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">{t['features']['realtime']['title']}</div>
                <div class="feature-desc">{t['features']['realtime']['desc']}</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">{t['features']['multilingual']['title']}</div>
                <div class="feature-desc">{t['features']['multilingual']['desc']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 聊天界面
    st.markdown(f"""
    <div class="chat-container">
        <div class="chat-header">
            🤖 {t['chat_title']}
        </div>
        <div class="chat-messages">
    """, unsafe_allow_html=True)
    
    # 初始化
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": t["welcome"]
        }]
    
    if "analyst" not in st.session_state:
        st.session_state.analyst = BilingualQuantGPTAnalyst(lang)
    
    # 显示消息
    for message in st.session_state.messages:
        display_message(message["content"], message["role"] == "user", lang)
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    # 用户输入
    placeholder_text = "输入您的分析需求..." if lang == "中文" else "Enter your analysis request..."
    user_input = st.chat_input(placeholder_text)
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🤖 分析中..." if lang == "中文" else "🤖 Analyzing..."):
            response = st.session_state.analyst.generate_response(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # 图表展示
    if 'analysis_data' in st.session_state and st.session_state.analysis_data:
        st.markdown("---")
        st.markdown(f'<div class="chart-title">📈 专业图表分析</div>' if lang == "中文" else f'<div class="chart-title">📈 Professional Chart Analysis</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(st.session_state.analysis_data))
        for i, stock in enumerate(st.session_state.analysis_data.keys()):
            with cols[i]:
                button_text = f"查看 {stock} 图表" if lang == "中文" else f"View {stock} Chart"
                if st.button(button_text, key=f"chart_{stock}", use_container_width=True):
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = create_openquant_style_chart(stock)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        # 语言选择
        st.markdown(f'<div class="sidebar-title">🌐 Language / 语言</div>', unsafe_allow_html=True)
        new_lang = st.selectbox("", ["中文", "English"], index=0 if lang == "中文" else 1)
        
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()
        
        # 快速分析
        st.markdown(f'<div class="sidebar-title">🚀 {t["quick_analysis"]}</div>', unsafe_allow_html=True)
        
        stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN"]
        selected_stock = st.selectbox(t["select_stock"], stocks)
        
        strategies = [t["trend_strategy"], t["mean_reversion"], t["momentum_strategy"]]
        selected_strategy = st.selectbox(t["select_strategy"], strategies)
        
        if st.button(t["start_analysis"], use_container_width=True, type="primary"):
            if lang == "中文":
                query = f"分析{selected_stock}的{selected_strategy}"
            else:
                strategy_map = {
                    "Trend Strategy": "trend strategy",
                    "Mean Reversion": "Bollinger Bands strategy", 
                    "Momentum Strategy": "RSI momentum strategy"
                }
                query = f"Analyze {selected_stock} {strategy_map.get(selected_strategy, 'strategy')}"
            
            st.session_state.messages.append({"role": "user", "content": query})
            response = st.session_state.analyst.generate_response(query)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # 示例查询
        st.markdown(f'<div class="sidebar-title">💡 {t["examples"]}</div>', unsafe_allow_html=True)
        
        if lang == "中文":
            examples = [
                "分析苹果公司的趋势策略",
                "特斯拉的布林带策略",
                "谷歌的RSI动量策略",
                "微软的双均线策略"
            ]
        else:
            examples = [
                "Analyze AAPL trend strategy",
                "Tesla Bollinger Bands strategy",
                "Google RSI momentum strategy",
                "Microsoft moving average strategy"
            ]
        
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                response = st.session_state.analyst.generate_response(example)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        # 清除历史
        if st.button(t["clear_history"], use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            if 'analysis_data' in st.session_state:
                del st.session_state.analysis_data
            st.rerun()

if __name__ == "__main__":
    main()
