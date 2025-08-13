import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import json
import re
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="QuantGPT - AI量化交易助手",
    page_icon="🤖",
    layout="wide"
)

# CSS样式
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #2b313e;
        flex-direction: row-reverse;
    }
    .chat-message.bot {
        background-color: #475063;
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
    }
    .chat-message .message {
        flex: 1;
    }
    .user .avatar {
        background-color: #1f77b4;
    }
    .bot .avatar {
        background-color: #f63366;
    }
    .strategy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-highlight {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

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
        
        data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
        
        data['Signal'] = 0
        data['Signal'][short_window:] = np.where(
            data['SMA_short'][short_window:] > data['SMA_long'][short_window:], 1, 0
        )
        data['Position'] = data['Signal'].diff()
        
        return data, f"使用{short_window}日和{long_window}日双均线策略，短期均线上穿长期均线时买入"
    
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
        
        return data, f"布林带均值回归策略，价格跌破下轨买入，涨破上轨卖出"
    
    def momentum_strategy(self, data, params):
        """动量策略"""
        rsi_window = params.get('rsi_window', 14)
        rsi_threshold = params.get('rsi_threshold', 70)
        
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
        
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
        
        return data, f"唐奇安通道突破策略，突破{window}日最高点买入，跌破最低点卖出"
    
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

# AI对话处理器
class QuantGPTProcessor:
    def __init__(self):
        self.strategy_engine = EnhancedStrategyEngine()
        self.risk_manager = RiskManager()
        self.backtest_engine = BacktestEngine()
        
    def parse_user_input(self, user_input):
        """解析用户输入"""
        user_input = user_input.lower()
        
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
            if keyword in user_input:
                detected_strategy = strategy
                break
        
        # 提取参数
        params = {}
        
        # 提取数字参数
        numbers = re.findall(r'\d+', user_input)
        if "天" in user_input or "日" in user_input:
            if len(numbers) >= 1:
                params['window'] = int(numbers[0])
                params['short_window'] = int(numbers[0])
            if len(numbers) >= 2:
                params['long_window'] = int(numbers[1])
        
        # 提取百分比参数
        percentage_matches = re.findall(r'(\d+(?:\.\d+)?)%', user_input)
        if percentage_matches:
            if "止损" in user_input:
                params['stop_loss'] = float(percentage_matches[0]) / 100
            elif "网格" in user_input:
                params['grid_size'] = float(percentage_matches[0]) / 100
        
        return {
            'stocks': stocks,
            'strategy': detected_strategy,
            'params': params,
            'original_input': user_input
        }
    
    def generate_response(self, parsed_input):
        """生成AI响应"""
        stocks = parsed_input['stocks']
        strategy = parsed_input['strategy']
        params = parsed_input['params']
        original_input = parsed_input['original_input']
        
        if not stocks:
            return "🤖 请告诉我您想分析的股票代码，例如：'帮我分析AAPL的趋势策略'"
        
        if not strategy:
            return f"🤖 我识别到股票代码：{', '.join(stocks)}。请告诉我您想使用什么策略？\n\n可选策略：\n• 趋势跟踪（双均线）\n• 均值回归（布林带）\n• 动量策略（RSI+MACD）\n• 突破策略（通道突破）\n• 网格交易\n• 配对交易"
        
        # 执行分析
        results = []
        for stock in stocks:
            try:
                # 获取数据
                data = yf.Ticker(stock).history(period="2y")
                if data.empty:
                    results.append(f"❌ 无法获取 {stock} 的数据")
                    continue
                
                # 添加技术指标
                data = self.add_technical_indicators(data)
                
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
    
    def add_technical_indicators(self, data):
        """添加技术指标"""
        # 趋势指标
        data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        
        # 动量指标
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
        
        # 波动率指标
        data['BB_upper'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
        data['BB_lower'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        
        return data
    
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
        result += "**回测指标：**\n"
        
        for metric, value in metrics.items():
            result += f"• {metric}: {value}\n"
        
        # 生成建议
        if "夏普比率" in metrics:
            sharpe = float(metrics["夏普比率"])
            if sharpe > 1.0:
                result += f"\n💡 **AI建议：** 该策略表现优秀，夏普比率达到{sharpe:.2f}，建议考虑实盘应用。"
            elif sharpe > 0.5:
                result += f"\n⚠️ **AI建议：** 该策略表现中等，建议优化参数或考虑组合策略。"
            else:
                result += f"\n❌ **AI建议：** 该策略表现较差，建议重新选择策略或调整参数。"
        
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
    
    # 创建子图
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{stock} 价格走势与交易信号', '技术指标', '策略收益'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # 价格线
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='收盘价', line=dict(color='blue')),
        row=1, col=1
    )
    
    # 买入信号
    buy_signals = data[data['Position'] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                      mode='markers', name='买入信号', 
                      marker=dict(color='green', size=10, symbol='triangle-up')),
            row=1, col=1
        )
    
    # 卖出信号
    sell_signals = data[data['Position'] == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                      mode='markers', name='卖出信号',
                      marker=dict(color='red', size=10, symbol='triangle-down')),
            row=1, col=1
        )
    
    # 技术指标
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red")
        fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green")
    
    # 策略收益对比
    fig.add_trace(
        go.Scatter(x=data.index, y=(data['Cumulative_Returns']-1)*100,
                  name='买入持有(%)', line=dict(color='gray')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=(data['Strategy_Cumulative']-1)*100,
                  name='策略收益(%)', line=dict(color='green')),
        row=3, col=1
    )
    
    fig.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# 主应用
def main():
    st.title("🤖 QuantGPT - AI量化交易助手")
    st.markdown("---")
    
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是QuantGPT，您的AI量化交易助手。\n\n我可以帮您：\n• 分析股票策略（趋势、均值回归、动量等）\n• 进行回测分析\n• 提供投资建议\n• 风险管理指导\n\n请告诉我您想分析什么股票？例如：\n'帮我分析AAPL的趋势策略'\n'用布林带策略分析TSLA'\n'GOOGL的20日双均线策略'"}
        ]
    
    if "processor" not in st.session_state:
        st.session_state.processor = QuantGPTProcessor()
    
    # 显示聊天历史
    for message in st.session_state.messages:
        display_message(message["content"], message["role"] == "user")
    
    # 用户输入
    user_input = st.chat_input("请输入您的量化交易问题...")
    
    if user_input:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message(user_input, True)
        
        # 处理用户输入
        with st.spinner("🤖 正在分析中..."):
            parsed_input = st.session_state.processor.parse_user_input(user_input)
            response = st.session_state.processor.generate_response(parsed_input)
        
        # 添加AI响应
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_message(response, False)
        
        # 如果有分析数据，显示图表选项
        if 'analysis_data' in st.session_state and st.session_state.analysis_data:
            st.markdown("---")
            st.subheader("📈 查看详细图表")
            
            cols = st.columns(len(st.session_state.analysis_data))
            for i, stock in enumerate(st.session_state.analysis_data.keys()):
                with cols[i]:
                    if st.button(f"📊 {stock} 图表", key=f"chart_{stock}"):
                        show_analysis_chart(stock)
        
        # 重新运行显示更新
        st.rerun()
    
    # 侧边栏功能
    with st.sidebar:
        st.title("🛠️ 功能菜单")
        
        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = [st.session_state.messages[0]]  # 保留欢迎消息
            if 'analysis_data' in st.session_state:
                del st.session_state.analysis_data
            st.rerun()
        
        if st.button("📋 示例问题"):
            examples = [
                "分析AAPL的趋势策略",
                "用20日和50日双均线分析TSLA",
                "GOOGL的RSI动量策略回测",
                "用2%止损分析MSFT突破策略",
                "分析NVDA的网格交易策略"
            ]
            st.write("💡 **示例问题：**")
            for example in examples:
                st.write(f"• {example}")
        
        st.markdown("---")
        st.markdown("**支持的策略：**")
        st.markdown("• 趋势跟踪（双均线）")
        st.markdown("• 均值回归（布林带）") 
        st.markdown("• 动量策略（RSI+MACD）")
        st.markdown("• 突破策略（通道突破）")
        st.markdown("• 网格交易")
        st.markdown("• 配对交易")
        
        st.markdown("---")
        st.markdown("**支持的参数：**")
        st.markdown("• 时间窗口（如：20天、50日）")
        st.markdown("• 止损百分比（如：5%止损）")
        st.markdown("• 网格大小（如：2%网格）")

if __name__ == "__main__":
    main()
