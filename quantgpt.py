import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import re
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
import warnings

# 尝试导入可选依赖
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.error("⚠️ yfinance未安装，数据获取功能不可用")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    st.warning("⚠️ ta库未安装，将使用简化的技术指标")

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="QuantGPT - AI量化交易助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 增强的CSS样式
st.markdown("""
<style>
    .main > div {
        padding: 2rem 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .chat-message:hover {
        transform: translateY(-2px);
    }
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        flex-direction: row-reverse;
    }
    .chat-message.bot {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .chat-message .avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        margin: 0 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    .chat-message .message {
        flex: 1;
        line-height: 1.6;
    }
    .strategy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-highlight {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .block-container {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 模拟数据生成器（当yfinance不可用时）
class MockDataGenerator:
    @staticmethod
    def generate_mock_data(symbol, period="2y"):
        """生成模拟股价数据"""
        end_date = datetime.now()
        if period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=180)
        
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(hash(symbol) % 1000)  # 确保同一股票的数据一致
        
        # 生成随机游走价格
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [100]  # 起始价格
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # 创建OHLC数据
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
        data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
        data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
        data['Volume'] = np.random.randint(1000000, 10000000, len(data))
        
        return data.dropna()

# 简化的技术指标计算（当ta不可用时）
class SimpleTechnicalIndicators:
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        return data.ewm(span=window).mean()
    
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
        return macd_line, signal_line

# 增强的策略引擎
class EnhancedStrategyEngine:
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
            data['SMA_short'] = SimpleTechnicalIndicators.sma(data['Close'], short_window)
            data['SMA_long'] = SimpleTechnicalIndicators.sma(data['Close'], long_window)
        
        data['Signal'] = 0
        data['Signal'][short_window:] = np.where(
            data['SMA_short'][short_window:] > data['SMA_long'][short_window:], 1, 0
        )
        data['Position'] = data['Signal'].diff()
        
        return data, f"使用{short_window}日和{long_window}日双均线策略，短期均线上穿长期均线时买入"
    
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
            data['Upper'], data['Lower'], data['SMA'] = SimpleTechnicalIndicators.bollinger_bands(
                data['Close'], window, std_dev
            )
        
        data['Signal'] = 0
        data['Signal'] = np.where(data['Close'] < data['Lower'], 1, 
                                np.where(data['Close'] > data['Upper'], -1, 0))
        data['Position'] = data['Signal'].diff()
        
        return data, f"布林带均值回归策略（窗口{window}，标准差{std_dev}），价格跌破下轨买入，涨破上轨卖出"
    
    def momentum_strategy(self, data, params):
        """动量策略"""
        rsi_window = params.get('rsi_window', 14)
        rsi_threshold = params.get('rsi_threshold', 70)
        
        if TA_AVAILABLE:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
            data['MACD'] = ta.trend.MACD(data['Close']).macd()
            data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
        else:
            data['RSI'] = SimpleTechnicalIndicators.rsi(data['Close'], rsi_window)
            data['MACD'], data['MACD_signal'] = SimpleTechnicalIndicators.macd(data['Close'])
        
        data['Signal'] = 0
        data['Signal'] = np.where(
            (data['RSI'] > rsi_threshold) & (data['MACD'] > data['MACD_signal']), 1,
            np.where((data['RSI'] < (100-rsi_threshold)) & (data['MACD'] < data['MACD_signal']), -1, 0)
        )
        data['Position'] = data['Signal'].diff()
        
        return data, f"RSI({rsi_window})和MACD组合动量策略，RSI超买超卖结合MACD信号"
    
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
        
        return data, f"唐奇安通道突破策略（窗口{window}），突破最高点买入，跌破最低点卖出"
    
    def grid_trading(self, data, params):
        """网格交易策略"""
        grid_size = params.get('grid_size', 0.02)
        
        data['Price_change'] = data['Close'].pct_change().cumsum()
        data['Grid_level'] = (data['Price_change'] / grid_size).round()
        data['Signal'] = -data['Grid_level'].diff()
        data['Position'] = data['Signal'].diff()
        
        return data, f"网格交易策略，网格间距{grid_size*100:.1f}%，价格上涨卖出，下跌买入"
    
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
        
        return data, f"统计套利策略，基于{window}日价差均值回归"

# 风险管理器
class RiskManager:
    @staticmethod
    def add_stop_loss(data, stop_loss_pct=0.05):
        """添加止损"""
        data['Stop_Loss_Signal'] = 0
        entry_price = None
        position = 0
        
        for i in range(len(data)):
            if data['Position'].iloc[i] == 1:
                entry_price = data['Close'].iloc[i]
                position = 1
            elif data['Position'].iloc[i] == -1:
                position = 0
                entry_price = None
            
            if position == 1 and entry_price:
                stop_price = entry_price * (1 - stop_loss_pct)
                if data['Close'].iloc[i] < stop_price:
                    data.iloc[i, data.columns.get_loc('Stop_Loss_Signal')] = -1
                    position = 0
                    entry_price = None
        
        return data
    
    @staticmethod
    def calculate_position_size(capital, risk_per_trade, price, stop_loss_pct):
        """计算仓位大小"""
        risk_amount = capital * risk_per_trade
        max_loss_per_share = price * stop_loss_pct
        position_size = risk_amount / max_loss_per_share if max_loss_per_share > 0 else 0
        return min(position_size, capital / price)

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
            "波动率": f"{volatility:.2%}"
        }

# AI对话处理器
class QuantGPTProcessor:
    def __init__(self):
        self.strategy_engine = EnhancedStrategyEngine()
        self.risk_manager = RiskManager()
        self.backtest_engine = BacktestEngine()
        
    def parse_user_input(self, user_input):
        """解析用户输入"""
        user_input_lower = user_input.lower()
        
        # 提取股票代码
        stock_pattern = r'\b[A-Z]{1,5}\b'
        stocks = re.findall(stock_pattern, user_input.upper())
        
        # 提取策略类型
        strategy_keywords = {
            "趋势": "趋势跟踪",
            "均线": "趋势跟踪", 
            "双均线": "趋势跟踪",
            "均值回归": "均值回归",
            "布林带": "均值回归",
            "动量": "动量策略",
            "rsi": "动量策略",
            "macd": "动量策略",
            "突破": "突破策略",
            "通道": "突破策略",
            "网格": "网格交易",
            "配对": "配对交易",
            "套利": "配对交易"
        }
        
        detected_strategy = None
        for keyword, strategy in strategy_keywords.items():
            if keyword in user_input_lower:
                detected_strategy = strategy
                break
        
        # 提取参数
        params = {}
        
        # 提取数字参数
        numbers = re.findall(r'\d+', user_input)
        if any(word in user_input_lower for word in ["天", "日", "period"]):
            if len(numbers) >= 1:
                params['window'] = int(numbers[0])
                params['short_window'] = int(numbers[0])
            if len(numbers) >= 2:
                params['long_window'] = int(numbers[1])
        
        # 提取百分比参数
        percentage_matches = re.findall(r'(\d+(?:\.\d+)?)%', user_input)
        if percentage_matches:
            if "止损" in user_input_lower:
                params['stop_loss'] = float(percentage_matches[0]) / 100
            elif "网格" in user_input_lower:
                params['grid_size'] = float(percentage_matches[0]) / 100
        
        return {
            'stocks': stocks,
            'strategy': detected_strategy,
            'params': params,
            'original_input': user_input
        }
    
    def get_stock_data(self, symbol, period="2y"):
        """获取股票数据"""
        if YFINANCE_AVAILABLE:
            try:
                data = yf.Ticker(symbol).history(period=period)
                if not data.empty:
                    return data
            except Exception as e:
                st.warning(f"获取{symbol}数据失败，使用模拟数据: {str(e)}")
        
        # 使用模拟数据
        return MockDataGenerator.generate_mock_data(symbol, period)
    
    def generate_response(self, parsed_input):
        """生成AI响应"""
        stocks = parsed_input['stocks']
        strategy = parsed_input['strategy']
        params = parsed_input['params']
        original_input = parsed_input['original_input']
        
        if not stocks:
            return "🤖 请告诉我您想分析的股票代码，例如：'帮我分析AAPL的趋势策略'"
        
        if not strategy:
            return f"🤖 我识别到股票代码：{', '.join(stocks)}。请告诉我您想使用什么策略？\n\n📊 **可选策略：**\n• 趋势跟踪（双均线）\n• 均值回归（布林带）\n• 动量策略（RSI+MACD）\n• 突破策略（通道突破）\n• 网格交易\n• 配对交易"
        
        # 执行分析
        results = []
        for stock in stocks:
            try:
                # 获取数据
                data = self.get_stock_data(stock)
                if data.empty:
                    results.append(f"❌ 无法获取 {stock} 的数据")
                    continue
                
                # 设置默认参数
                default_params = self.get_default_params(strategy)
                default_params.update(params)
                
                # 运行策略
                strategy_data, description = self.strategy_engine.strategies[strategy](data.copy(), default_params)
                
                # 添加风险管理
                if 'stop_loss' in params:
                    strategy_data = self.risk_manager.add_stop_loss(strategy_data, params['stop_loss'])
                
                # 运行回测
                backtest_data = self.backtest_engine.run_backtest(strategy_data)
                
                # 计算指标
                metrics = self.backtest_engine.calculate_metrics(backtest_data)
                
                # 格式化结果
                result = self.format_analysis_result(stock, strategy, description, metrics, default_params)
                results.append(result)
                
                # 存储数据用于可视化
                if 'analysis_data' not in st.session_state:
                    st.session_state.analysis_data = {}
                st.session_state.analysis_data[stock] = {
                    'data': backtest_data,
                    'strategy': strategy,
                    'params': default_params
                }
                
            except Exception as e:
                results.append(f"❌ 分析 {stock} 时出错：{str(e)}")
        
        return "\n\n".join(results)
    
    def get_default_params(self, strategy):
        """获取策略默认参数"""
        defaults = {
            "趋势跟踪": {'short_window': 20, 'long_window': 50},
            "均值回归": {'window': 20, 'std_dev': 2.0},
            "动量策略": {'rsi_window': 14, 'rsi_threshold': 70},
            "突破策略": {'window': 20},
            "网格交易": {'grid_size': 0.02},
            "配对交易": {'window': 30}
        }
        return defaults.get(strategy, {})
    
    def format_analysis_result(self, stock, strategy, description, metrics, params):
        """格式化分析结果"""
        result = f"## 📊 {stock} - {strategy}分析结果\n\n"
        result += f"**策略说明：** {description}\n\n"
        result += f"**参数设置：** {json.dumps(params, ensure_ascii=False)}\n\n"
        result += "**📈 回测指标：**\n"
        
        for metric, value in metrics.items():
            result += f"• **{metric}**: {value}\n"
        
        # 生成AI建议
        try:
            sharpe = float(metrics["夏普比率"])
            total_return = float(metrics["总收益率"].rstrip('%')) / 100
            max_drawdown = float(metrics["最大回撤"].rstrip('%')) / 100
            
            result += f"\n💡 **AI智能建议：**\n"
            
            if sharpe > 1.5:
                result += f"🟢 **优秀策略** - 夏普比率{sharpe:.2f}表现卓越，建议重点考虑实盘应用\n"
            elif sharpe > 1.0:
                result += f"🟡 **良好策略** - 夏普比率{sharpe:.2f}表现良好，可考虑适量配置\n"
            elif sharpe > 0.5:
                result += f"🟠 **一般策略** - 表现中等，建议优化参数或组合其他策略\n"
            else:
                result += f"🔴 **需要改进** - 表现较差，建议重新选择策略或大幅调整参数\n"
            
            if max_drawdown < -0.2:
                result += f"⚠️ **风险警告** - 最大回撤{max_drawdown:.1%}较大，请注意风险控制\n"
            
            if total_return > 0.2:
                result += f"📈 **收益亮点** - 总收益率{total_return:.1%}表现不错\n"
                
        except (ValueError, KeyError):
            result += f"\n💡 **AI建议：** 策略分析完成，请查看具体指标并结合市场环境判断。"
        
        return result

# 聊天消息显示函数
def display_message(message, is_user=False):
    """显示聊天消息"""
    message_class = "user" if is_user else "bot"
    avatar = "👤" if is_user else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="avatar">{avatar}</div>
        <div class="message">{message}</div>
    </div>
    """, unsafe_allow_html=True)

# 可视化函数
def show_analysis_chart(stock):
    """显示分析图表"""
    if 'analysis_data' not in st.session_state or stock not in st.session_state.analysis_data:
        st.error(f"没有 {stock} 的分析数据")
        return
    
    data = st.session_state.analysis_data[stock]['data']
    strategy = st.session_state.analysis_data[stock]['strategy']
    params = st.session_state.analysis_data[stock]['params']
    
    # 创建子图
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{stock} 价格走势与交易信号', 
            '技术指标', 
            '策略收益对比',
            '回撤分析'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # 价格线和移动平均线
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='收盘价', 
                  line=dict(color='#2E86AB', width=2)),
        row=
