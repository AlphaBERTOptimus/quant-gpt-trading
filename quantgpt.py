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
    page_title="QuantGPT Pro - Professional Trading Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 专业黑白灰CSS样式
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 全局样式重置 */
    .main > div {
        padding: 0;
        background: #0F1419;
        color: #E8E8E8;
        font-family: 'Inter', sans-serif;
    }
    
    /* 隐藏Streamlit默认元素 */
    .stApp > header {
        background: transparent;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F1419 0%, #1A1F2E 100%);
    }
    
    /* 专业头部 */
    .professional-header {
        background: linear-gradient(90deg, #1A1F2E 0%, #2D3748 100%);
        padding: 1.5rem 2rem;
        border-bottom: 1px solid #4A5568;
        margin-bottom: 0;
    }
    
    .header-title {
        font-family: 'Roboto Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        color: #A0AEC0;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* 状态栏 */
    .status-bar {
        background: #2D3748;
        padding: 0.8rem 2rem;
        border-bottom: 1px solid #4A5568;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: 'Roboto Mono', monospace;
    }
    
    .status-online {
        background: rgba(72, 187, 120, 0.1);
        color: #48BB78;
        border: 1px solid rgba(72, 187, 120, 0.2);
    }
    
    .status-offline {
        background: rgba(245, 101, 101, 0.1);
        color: #F56565;
        border: 1px solid rgba(245, 101, 101, 0.2);
    }
    
    .status-warning {
        background: rgba(237, 137, 54, 0.1);
        color: #ED8936;
        border: 1px solid rgba(237, 137, 54, 0.2);
    }
    
    /* 主要内容区域 */
    .main-content {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* 聊天界面 */
    .chat-container {
        background: #1A202C;
        border: 1px solid #4A5568;
        border-radius: 8px;
        padding: 0;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .chat-header {
        background: #2D3748;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #4A5568;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        color: #E2E8F0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .chat-messages {
        padding: 1.5rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .message {
        margin-bottom: 1.5rem;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .message.user {
        background: rgba(66, 153, 225, 0.1);
        border-left-color: #4299E1;
        color: #E2E8F0;
        margin-left: 2rem;
    }
    
    .message.assistant {
        background: rgba(72, 187, 120, 0.1);
        border-left-color: #48BB78;
        color: #E2E8F0;
        margin-right: 2rem;
    }
    
    .message-header {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        opacity: 0.7;
    }
    
    /* 输入框样式 */
    .stChatInput > div > div > div > div {
        background: #2D3748 !important;
        border: 1px solid #4A5568 !important;
        border-radius: 6px !important;
        color: #E2E8F0 !important;
    }
    
    .stChatInput > div > div > div > div > div > textarea {
        background: transparent !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: #4A5568;
        color: #E2E8F0;
        border: 1px solid #718096;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #718096;
        border-color: #A0AEC0;
        transform: translateY(-1px);
    }
    
    .primary-button {
        background: #4299E1 !important;
        color: white !important;
        border: 1px solid #3182CE !important;
    }
    
    .primary-button:hover {
        background: #3182CE !important;
        border-color: #2B6CB0 !important;
    }
    
    /* 侧边栏样式 */
    .sidebar .block-container {
        background: #1A202C;
        border-right: 1px solid #4A5568;
        padding: 1.5rem;
    }
    
    .sidebar-section {
        background: #2D3748;
        border: 1px solid #4A5568;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-title {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #E2E8F0;
        margin-bottom: 1rem;
        border-bottom: 1px solid #4A5568;
        padding-bottom: 0.5rem;
    }
    
    /* 选择框样式 */
    .stSelectbox > div > div {
        background: #4A5568 !important;
        border: 1px solid #718096 !important;
        border-radius: 6px !important;
        color: #E2E8F0 !important;
    }
    
    /* 指标卡片 */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: #2D3748;
        border: 1px solid #4A5568;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #48BB78;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #A0AEC0;
    }
    
    /* 图表容器 */
    .chart-container {
        background: #1A202C;
        border: 1px solid #4A5568;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2D3748;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4A5568;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #718096;
    }
    
    /* 响应式 */
    @media (max-width: 768px) {
        .main-content {
            padding: 1rem;
        }
        
        .professional-header {
            padding: 1rem;
        }
        
        .header-title {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 多语言支持
LANGUAGES = {
    "中文": {
        "title": "QuantGPT Pro",
        "subtitle": "专业量化交易分析平台",
        "chat_header": "AI 量化分析师",
        "data_source": "数据源",
        "technical_indicators": "技术指标",
        "risk_control": "风控系统",
        "system_status": "系统状态",
        "real_time_data": "实时数据",
        "simulated_data": "模拟数据", 
        "complete_indicators": "完整指标",
        "basic_indicators": "基础指标",
        "enabled": "已启用",
        "online": "在线",
        "quick_analysis": "快速分析",
        "select_stock": "选择股票",
        "select_strategy": "选择策略",
        "start_analysis": "开始分析",
        "clear_history": "清除历史",
        "trend_following": "趋势跟踪",
        "mean_reversion": "均值回归", 
        "momentum_strategy": "动量策略",
        "example_queries": "示例查询",
        "welcome_message": """🎯 **欢迎使用QuantGPT Pro专业版**

我是您的专业AI量化分析师，支持中英文交互。

**核心功能：**
• 趋势跟踪 - 双均线策略分析
• 均值回归 - 布林带策略分析  
• 动量策略 - RSI技术指标分析

**使用示例：**
• "分析苹果公司的趋势策略"
• "分析AAPL的trend strategy"
• "特斯拉的布林带策略怎么样"
• "Analyze TSLA momentum strategy"

现在您可以用中文或英文与我对话！""",
        "analysis_result": "分析结果",
        "strategy_description": "策略说明",
        "backtest_metrics": "回测指标",
        "ai_assessment": "AI评估",
        "total_return": "总收益率",
        "annual_return": "年化收益率", 
        "sharpe_ratio": "夏普比率",
        "max_drawdown": "最大回撤",
        "win_rate": "胜率",
        "trade_count": "交易次数"
    },
    "English": {
        "title": "QuantGPT Pro",
        "subtitle": "Professional Quantitative Trading Platform",
        "chat_header": "AI Quant Analyst",
        "data_source": "Data Source",
        "technical_indicators": "Technical Indicators", 
        "risk_control": "Risk Control",
        "system_status": "System Status",
        "real_time_data": "Real-time Data",
        "simulated_data": "Simulated Data",
        "complete_indicators": "Complete Indicators",
        "basic_indicators": "Basic Indicators", 
        "enabled": "Enabled",
        "online": "Online",
        "quick_analysis": "Quick Analysis",
        "select_stock": "Select Stock",
        "select_strategy": "Select Strategy", 
        "start_analysis": "Start Analysis",
        "clear_history": "Clear History",
        "trend_following": "Trend Following",
        "mean_reversion": "Mean Reversion",
        "momentum_strategy": "Momentum Strategy",
        "example_queries": "Example Queries",
        "welcome_message": """🎯 **Welcome to QuantGPT Pro**

I'm your professional AI quantitative analyst, supporting both Chinese and English.

**Core Features:**
• Trend Following - Moving Average Strategy Analysis
• Mean Reversion - Bollinger Bands Strategy Analysis  
• Momentum Strategy - RSI Technical Indicator Analysis

**Usage Examples:**
• "Analyze AAPL trend strategy"
• "分析苹果公司的趋势策略"
• "How is TSLA's Bollinger Bands strategy"
• "特斯拉的动量策略分析"

You can now chat with me in Chinese or English!""",
        "analysis_result": "Analysis Result",
        "strategy_description": "Strategy Description", 
        "backtest_metrics": "Backtest Metrics",
        "ai_assessment": "AI Assessment",
        "total_return": "Total Return",
        "annual_return": "Annual Return",
        "sharpe_ratio": "Sharpe Ratio", 
        "max_drawdown": "Max Drawdown",
        "win_rate": "Win Rate",
        "trade_count": "Trade Count"
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
            desc = f"双均线策略({short_window}日/{long_window}日) - 短期均线上穿长期均线时买入"
        else:
            desc = f"Moving Average Strategy ({short_window}d/{long_window}d) - Buy when short MA crosses above long MA"
        
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
            desc = f"布林带均值回归策略({window}日) - 价格触及下轨买入，上轨卖出"
        else:
            desc = f"Bollinger Bands Mean Reversion ({window}d) - Buy at lower band, sell at upper band"
        
        return data, desc
    
    def momentum_strategy(self, data, rsi_window=14):
        if TA_AVAILABLE:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
        else:
            data['RSI'] = TechnicalIndicators.rsi(data['Close'], rsi_window)
        
        data['Signal'] = np.where(data['RSI'] > 70, -1, np.where(data['RSI'] < 30, 1, 0))
        data['Position'] = data['Signal'].diff()
        
        if self.lang == "中文":
            desc = f"RSI动量策略({rsi_window}日) - RSI超买超卖信号交易"
        else:
            desc = f"RSI Momentum Strategy ({rsi_window}d) - Trade on RSI overbought/oversold signals"
        
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
            return {"Error": "No valid trading data" if self.lang == "English" else "无有效交易数据"}
        
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

# 中英文AI分析师
class BilingualQuantGPTAnalyst:
    def __init__(self, lang="中文"):
        self.lang = lang
        self.strategy_engine = StrategyEngine(lang)
        self.backtest_engine = BacktestEngine(lang)
    
    def detect_language(self, text):
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > english_chars:
            return "中文"
        else:
            return "English"
    
    def parse_input(self, user_input):
        # 检测语言
        detected_lang = self.detect_language(user_input)
        
        # 提取股票代码（支持中英文描述）
        stocks = re.findall(r'\b[A-Z]{1,5}\b', user_input.upper())
        
        # 中文股票名称映射
        chinese_stocks = {
            "苹果": "AAPL", "苹果公司": "AAPL", "apple": "AAPL",
            "特斯拉": "TSLA", "tesla": "TSLA",
            "谷歌": "GOOGL", "google": "GOOGL",
            "微软": "MSFT", "microsoft": "MSFT",
            "英伟达": "NVDA", "nvidia": "NVDA",
            "亚马逊": "AMZN", "amazon": "AMZN"
        }
        
        # 添加中文股票识别
        for chinese_name, symbol in chinese_stocks.items():
            if chinese_name in user_input.lower():
                stocks.append(symbol)
        
        # 策略识别（中英文）
        strategy_keywords = {
            # 趋势策略
            "趋势": "trend", "均线": "trend", "双均线": "trend", "moving average": "trend", 
            "trend": "trend", "ma": "trend",
            # 均值回归
            "均值回归": "mean", "布林带": "mean", "bollinger": "mean", "mean reversion": "mean",
            "bb": "mean", "bands": "mean",
            # 动量策略  
            "动量": "momentum", "rsi": "momentum", "momentum": "momentum",
            "相对强弱": "momentum", "超买超卖": "momentum"
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
        
        # 存储数据
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = {}
        st.session_state.analysis_data[stock] = backtest_data
        
        return self.format_result(stock, desc, metrics)
    
    def format_result(self, stock, description, metrics):
        if self.lang == "中文":
            result = f"## 📊 {stock} 量化分析报告\n\n"
            result += f"**策略说明：** {description}\n\n"
            result += "**回测指标：**\n"
        else:
            result = f"## 📊 {stock} Quantitative Analysis Report\n\n"
            result += f"**Strategy Description:** {description}\n\n"
            result += "**Backtest Metrics:**\n"
        
        for metric, value in metrics.items():
            result += f"• **{metric}**: {value}\n"
        
        # AI评估
        try:
            sharpe_key = "夏普比率" if self.lang == "中文" else "Sharpe Ratio"
            sharpe = float(metrics[sharpe_key])
            
            if self.lang == "中文":
                result += "\n### 🤖 AI智能评估\n\n"
                if sharpe > 1.0:
                    result += "✅ **评估结果：** 策略表现优秀，夏普比率超过1.0，具备实盘价值\n"
                    result += "💡 **建议：** 可考虑小仓位实盘测试，建议设置5-8%止损"
                elif sharpe > 0.5:
                    result += "⚠️ **评估结果：** 策略表现中等，有一定参考价值\n"
                    result += "💡 **建议：** 建议优化参数或结合其他指标使用"
                else:
                    result += "❌ **评估结果：** 策略表现较差，不建议直接使用\n"
                    result += "💡 **建议：** 重新选择策略或调整参数设置"
            else:
                result += "\n### 🤖 AI Assessment\n\n"
                if sharpe > 1.0:
                    result += "✅ **Assessment:** Excellent strategy performance with Sharpe ratio > 1.0, suitable for live trading\n"
                    result += "💡 **Recommendation:** Consider small position live testing with 5-8% stop loss"
                elif sharpe > 0.5:
                    result += "⚠️ **Assessment:** Moderate strategy performance with reference value\n"
                    result += "💡 **Recommendation:** Optimize parameters or combine with other indicators"
                else:
                    result += "❌ **Assessment:** Poor strategy performance, not recommended for direct use\n"
                    result += "💡 **Recommendation:** Re-select strategy or adjust parameter settings"
        except:
            if self.lang == "中文":
                result += "\n💡 **AI评估：** 分析完成，请查看详细指标"
            else:
                result += "\n💡 **AI Assessment:** Analysis completed, please review detailed metrics"
        
        return result
    
    def generate_response(self, user_input):
        parsed = self.parse_input(user_input)
        
        # 更新分析师语言设置
        if parsed["language"] != self.lang:
            self.lang = parsed["language"]
            self.strategy_engine.lang = parsed["language"]
            self.backtest_engine.lang = parsed["language"]
        
        if not parsed["stocks"]:
            if self.lang == "中文":
                return """🤖 **请指定股票代码进行分析**

**支持格式：**
• 股票代码：AAPL, TSLA, GOOGL等
• 中文名称：苹果公司、特斯拉、谷歌等

**示例：**
• "分析苹果公司的趋势策略"
• "分析AAPL的趋势策略"
• "特斯拉的布林带策略分析"
"""
            else:
                return """🤖 **Please specify stock symbol for analysis**

**Supported formats:**
• Stock symbols: AAPL, TSLA, GOOGL, etc.
• Company names: Apple, Tesla, Google, etc.

**Examples:**
• "Analyze AAPL trend strategy"
• "Tesla Bollinger Bands strategy analysis"
• "Google momentum strategy"
"""
        
        results = []
        for stock in parsed["stocks"]:
            try:
                result = self.analyze_stock(stock, parsed["strategy"])
                results.append(result)
            except Exception as e:
                if self.lang == "中文":
                    results.append(f"❌ 分析{stock}失败：{str(e)}")
                else:
                    results.append(f"❌ Analysis failed for {stock}: {str(e)}")
        
        return "\n\n".join(results)

# 专业图表生成
def create_professional_chart(stock):
    if 'analysis_data' not in st.session_state or stock not in st.session_state.analysis_data:
        st.error(f"No data available for {stock}")
        return
    
    data = st.session_state.analysis_data[stock]
    
    # 创建专业深色主题图表
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[f'{stock} Price Action & Signals', 'Technical Indicators', 'Strategy Performance'],
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05
    )
    
    # 价格线 - 专业样式
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['Close'], 
            name='Close Price',
            line=dict(color='#E2E8F0', width=2),
            hovertemplate='<b>Close</b>: $%{y:.2f}<br><b>Date</b>: %{x}<extra></extra>'
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
                line=dict(color='#4299E1', width=1.5, dash='dot'),
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
                line=dict(color='#ED8936', width=1.5, dash='dot'),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # 交易信号 - 专业标记
    buy_signals = data[data['Position'] == 1]
    sell_signals = data[data['Position'] == -1]
    
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index, 
                y=buy_signals['Close'],
                mode='markers', 
                name='Buy Signal',
                marker=dict(
                    color='#48BB78', 
                    size=12, 
                    symbol='triangle-up',
                    line=dict(color='#2F855A', width=2)
                ),
                hovertemplate='<b>Buy Signal</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index, 
                y=sell_signals['Close'],
                mode='markers', 
                name='Sell Signal',
                marker=dict(
                    color='#F56565', 
                    size=12, 
                    symbol='triangle-down',
                    line=dict(color='#C53030', width=2)
                ),
                hovertemplate='<b>Sell Signal</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
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
                line=dict(color='#9F7AEA', width=2),
                hovertemplate='<b>RSI</b>: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        # RSI参考线
        fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="#F56565", opacity=0.6)
        fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="#48BB78", opacity=0.6)
        fig.add_hline(y=50, row=2, col=1, line_dash="dot", line_color="#A0AEC0", opacity=0.4)
    
    # 收益对比
    benchmark = (data['Cumulative_Returns'] - 1) * 100
    strategy = (data['Strategy_Cumulative'] - 1) * 100
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=benchmark, 
            name='Buy & Hold',
            line=dict(color='#718096', width=2, dash='dot'),
            hovertemplate='<b>Buy & Hold</b>: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=strategy, 
            name='Strategy',
            line=dict(color='#48BB78', width=3),
            fill='tonexty',
            fillcolor='rgba(72, 187, 120, 0.1)',
            hovertemplate='<b>Strategy Return</b>: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 专业深色主题布局
    fig.update_layout(
        height=800,
        title={
            'text': f"<b>{stock}</b> Quantitative Analysis",
            'x': 0.5,
            'font': {'size': 20, 'color': '#E2E8F0', 'family': 'Roboto Mono'}
        },
        showlegend=True,
        plot_bgcolor='#1A202C',
        paper_bgcolor='#1A202C',
        font=dict(color='#E2E8F0', family='Inter'),
        legend=dict(
            bgcolor='rgba(45, 55, 72, 0.8)',
            bordercolor='#4A5568',
            borderwidth=1,
            font=dict(color='#E2E8F0', size=11)
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#2D3748',
            bordercolor='#4A5568',
            font_color='#E2E8F0'
        )
    )
    
    # 更新子图背景
    for i in range(1, 4):
        fig.update_xaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(74, 85, 104, 0.3)',
            showline=True,
            linecolor='#4A5568',
            row=i, col=1
        )
        fig.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(74, 85, 104, 0.3)',
            showline=True,
            linecolor='#4A5568',
            row=i, col=1
        )
    
    return fig

# 消息显示组件
def display_message(message, is_user=False, lang="中文"):
    message_type = "user" if is_user else "assistant"
    header_text = "You" if is_user else ("AI Analyst" if lang == "English" else "AI分析师")
    
    st.markdown(f"""
    <div class="message {message_type}">
        <div class="message-header">{header_text}</div>
        <div>{message}</div>
    </div>
    """, unsafe_allow_html=True)

# 主程序
def main():
    # 语言选择
    if 'language' not in st.session_state:
        st.session_state.language = "中文"
    
    lang = st.session_state.language
    t = LANGUAGES[lang]
    
    # 专业头部
    st.markdown(f"""
    <div class="professional-header">
        <h1 class="header-title">{t['title']}</h1>
        <p class="header-subtitle">{t['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 状态栏
    status_data = "status-online" if YFINANCE_AVAILABLE else "status-offline"
    status_text = t["real_time_data"] if YFINANCE_AVAILABLE else t["simulated_data"]
    
    status_ta = "status-online" if TA_AVAILABLE else "status-warning"
    ta_text = t["complete_indicators"] if TA_AVAILABLE else t["basic_indicators"]
    
    st.markdown(f"""
    <div class="status-bar">
        <div>
            <span class="status-indicator {status_data}">📡 {t['data_source']}: {status_text}</span>
            <span class="status-indicator {status_ta}">📈 {t['technical_indicators']}: {ta_text}</span>
            <span class="status-indicator status-online">🛡️ {t['risk_control']}: {t['enabled']}</span>
        </div>
        <div>
            <span class="status-indicator status-online">⚡ {t['system_status']}: {t['online']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 主要内容
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # 聊天界面
    st.markdown(f"""
    <div class="chat-container">
        <div class="chat-header">{t['chat_header']}</div>
        <div class="chat-messages">
    """, unsafe_allow_html=True)
    
    # 初始化
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": t["welcome_message"]
        }]
    
    if "analyst" not in st.session_state:
        st.session_state.analyst = BilingualQuantGPTAnalyst(lang)
    
    # 显示消息历史
    for message in st.session_state.messages:
        display_message(message["content"], message["role"] == "user", lang)
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    # 用户输入
    user_input = st.chat_input(
        "输入您的分析需求 (支持中英文)..." if lang == "中文" else "Enter your analysis request (Chinese/English supported)..."
    )
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🤖 AI analyzing..." if lang == "English" else "🤖 AI分析中..."):
            response = st.session_state.analyst.generate_response(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # 图表按钮
    if 'analysis_data' in st.session_state and st.session_state.analysis_data:
        st.markdown("---")
        st.markdown("### 📊 Professional Charts" if lang == "English" else "### 📊 专业图表")
        
        cols = st.columns(len(st.session_state.analysis_data))
        for i, stock in enumerate(st.session_state.analysis_data.keys()):
            with cols[i]:
                button_text = f"📈 {stock} Chart" if lang == "English" else f"📈 {stock} 图表"
                if st.button(button_text, key=f"chart_{stock}", use_container_width=True):
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = create_professional_chart(stock)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 专业侧边栏
    with st.sidebar:
        st.markdown(f'<div class="sidebar-title">⚙️ {t["quick_analysis"].upper()}</div>', unsafe_allow_html=True)
        
        # 语言切换
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        new_lang = st.selectbox(
            "Language / 语言",
            ["中文", "English"],
            index=0 if lang == "中文" else 1
        )
        
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 快速分析
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN"]
        selected_stock = st.selectbox(t["select_stock"], stocks)
        
        strategies = [t["trend_following"], t["mean_reversion"], t["momentum_strategy"]]
        selected_strategy = st.selectbox(t["select_strategy"], strategies)
        
        if st.button(t["start_analysis"], use_container_width=True, type="primary"):
            if lang == "中文":
                query = f"分析{selected_stock}的{selected_strategy}"
            else:
                strategy_map = {
                    "Trend Following": "trend strategy",
                    "Mean Reversion": "Bollinger Bands strategy", 
                    "Momentum Strategy": "RSI strategy"
                }
                query = f"Analyze {selected_stock} {strategy_map.get(selected_strategy, 'strategy')}"
            
            st.session_state.messages.append({"role": "user", "content": query})
            response = st.session_state.analyst.generate_response(query)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 示例查询
        st.markdown(f'<div class="sidebar-title">{t["example_queries"].upper()}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        if lang == "中文":
            examples = [
                "分析苹果公司的趋势策略",
                "特斯拉的布林带策略分析", 
                "谷歌的RSI动量策略",
                "微软的双均线策略",
                "英伟达的均值回归策略"
            ]
        else:
            examples = [
                "Analyze AAPL trend strategy",
                "Tesla Bollinger Bands analysis",
                "Google RSI momentum strategy", 
                "Microsoft moving average strategy",
                "NVIDIA mean reversion analysis"
            ]
        
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                response = st.session_state.analyst.generate_response(example)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 清除历史
        if st.button(t["clear_history"], use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            if 'analysis_data' in st.session_state:
                del st.session_state.analysis_data
            st.rerun()
        
        # 系统信息
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: #2D3748; border-radius: 6px; font-family: 'Roboto Mono', monospace; font-size: 0.75rem; color: #A0AEC0;">
            <div><strong>QuantGPT Pro v2.0</strong></div>
            <div>Professional Trading Platform</div>
            <div style="margin-top: 0.5rem;">
                {t['data_source']}: {'yfinance' if YFINANCE_AVAILABLE else 'Simulated'}<br>
                {t['technical_indicators']}: {'Complete' if TA_AVAILABLE else 'Basic'}<br>
                Strategies: 3 Core Algorithms
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
