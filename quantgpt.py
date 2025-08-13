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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="QuantGPT Contest - AI竞赛量化交易助手",
    page_icon="🏆",
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
    .contest-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .strategy-rank {
        background-color: #f0f2f6;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 4px solid #1f77b4;
    }
    .winner-strategy {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        border-left: 4px solid #ff6b35;
        color: #333;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ContestTrade核心组件
class StrategyAgent:
    """单个策略智能体"""
    def __init__(self, name, strategy_func, description):
        self.name = name
        self.strategy_func = strategy_func
        self.description = description
        self.performance_history = []
        self.current_weight = 1.0
        self.total_trades = 0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
    
    def execute_strategy(self, data, params):
        """执行策略并记录表现"""
        try:
            strategy_data, explanation = self.strategy_func(data.copy(), params)
            return strategy_data, explanation, True
        except Exception as e:
            return None, f"策略执行失败: {str(e)}", False
    
    def update_performance(self, returns, sharpe, max_dd, win_rate):
        """更新表现指标"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'returns': returns,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        })
        self.sharpe_ratio = sharpe
        self.max_drawdown = max_dd
        self.win_rate = win_rate

class PerformancePredictor:
    """策略表现预测器"""
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, agent):
        """提取策略特征"""
        if len(agent.performance_history) < 3:
            return np.array([0.5, 0.0, 0.0, 0.0, 0.0])  # 默认特征
        
        recent_history = agent.performance_history[-5:]  # 最近5次表现
        
        features = []
        # 最近表现趋势
        recent_sharpes = [h['sharpe'] for h in recent_history]
        features.append(np.mean(recent_sharpes))  # 平均夏普比率
        features.append(np.std(recent_sharpes))   # 夏普比率稳定性
        
        # 表现动量
        if len(recent_sharpes) >= 2:
            features.append(recent_sharpes[-1] - recent_sharpes[-2])  # 动量
        else:
            features.append(0.0)
        
        # 其他指标
        recent_returns = [h['returns'] for h in recent_history]
        features.append(np.mean(recent_returns))  # 平均收益
        features.append(agent.win_rate)  # 胜率
        
        return np.array(features)
    
    def train(self, agents_history):
        """训练预测模型"""
        if len(agents_history) < 10:  # 需要足够的历史数据
            return False
        
        X, y = [], []
        for agent_data in agents_history:
            features = agent_data['features']
            next_performance = agent_data['next_performance']
            X.append(features)
            y.append(next_performance)
        
        X = np.array(X)
        y = np.array(y)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return True
    
    def predict_performance(self, agent):
        """预测策略未来表现"""
        if not self.is_trained:
            # 基于历史平均的简单预测
            if len(agent.performance_history) > 0:
                recent_performance = agent.performance_history[-3:]
                avg_sharpe = np.mean([h['sharpe'] for h in recent_performance])
                return max(0.1, avg_sharpe)  # 最小权重0.1
            return 0.5  # 默认预测
        
        features = self.extract_features(agent).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        return max(0.1, prediction)  # 确保最小权重

class ContestEngine:
    """竞赛引擎 - ContestTrade核心实现"""
    def __init__(self):
        self.agents = {}
        self.predictor = PerformancePredictor()
        self.contest_history = []
        self.current_market_regime = "Normal"
        
    def register_agent(self, agent):
        """注册策略智能体"""
        self.agents[agent.name] = agent
    
    def quantify_performance(self, data_results):
        """量化阶段：评估所有策略的历史表现"""
        performance_scores = {}
        
        for agent_name, agent in self.agents.items():
            if agent_name in data_results:
                backtest_data = data_results[agent_name]['backtest_data']
                metrics = data_results[agent_name]['metrics']
                
                # 提取关键指标
                try:
                    sharpe = float(metrics['夏普比率'])
                    total_return = float(metrics['总收益率'].replace('%', '')) / 100
                    max_dd = abs(float(metrics['最大回撤'].replace('%', ''))) / 100
                    win_rate = float(metrics['胜率'].replace('%', '')) / 100
                    
                    # 更新智能体表现
                    agent.update_performance(total_return, sharpe, max_dd, win_rate)
                    
                    # 综合得分（可自定义权重）
                    score = (sharpe * 0.4 + total_return * 0.3 + 
                           (1 - max_dd) * 0.2 + win_rate * 0.1)
                    performance_scores[agent_name] = max(0.1, score)
                    
                except:
                    performance_scores[agent_name] = 0.5  # 默认分数
            else:
                performance_scores[agent_name] = 0.5
        
        return performance_scores
    
    def predict_future_performance(self):
        """预测阶段：预测未来表现"""
        predictions = {}
        
        for agent_name, agent in self.agents.items():
            predicted_score = self.predictor.predict_performance(agent)
            predictions[agent_name] = predicted_score
        
        return predictions
    
    def allocate_resources(self, current_scores, predicted_scores):
        """分配阶段：基于预测分配权重"""
        # 结合当前表现和预测表现
        combined_scores = {}
        
        for agent_name in self.agents.keys():
            current = current_scores.get(agent_name, 0.5)
            predicted = predicted_scores.get(agent_name, 0.5)
            
            # 权重组合：70%历史表现 + 30%预测表现
            combined_score = current * 0.7 + predicted * 0.3
            combined_scores[agent_name] = combined_score
        
        # 归一化权重
        total_score = sum(combined_scores.values())
        if total_score > 0:
            weights = {name: score/total_score for name, score in combined_scores.items()}
        else:
            # 均等权重作为后备
            n = len(self.agents)
            weights = {name: 1.0/n for name in self.agents.keys()}
        
        return weights, combined_scores
    
    def run_contest(self, data_results):
        """运行完整的竞赛周期"""
        # 阶段1：量化表现
        current_scores = self.quantify_performance(data_results)
        
        # 阶段2：预测未来
        predicted_scores = self.predict_future_performance()
        
        # 阶段3：分配资源
        final_weights, combined_scores = self.allocate_resources(current_scores, predicted_scores)
        
        # 记录竞赛历史
        contest_result = {
            'timestamp': datetime.now(),
            'current_scores': current_scores,
            'predicted_scores': predicted_scores,
            'final_weights': final_weights,
            'combined_scores': combined_scores
        }
        self.contest_history.append(contest_result)
        
        return contest_result
    
    def get_strategy_rankings(self, contest_result):
        """获取策略排名"""
        rankings = []
        for agent_name, score in contest_result['combined_scores'].items():
            agent = self.agents[agent_name]
            rankings.append({
                'name': agent_name,
                'score': score,
                'weight': contest_result['final_weights'][agent_name],
                'sharpe': agent.sharpe_ratio,
                'win_rate': agent.win_rate,
                'description': agent.description
            })
        
        # 按综合得分排序
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return rankings

# 增强的策略引擎（集成ContestTrade）
class EnhancedStrategyEngine:
    def __init__(self):
        # 定义所有策略
        self.strategy_functions = {
            "趋势跟踪": self.trend_following,
            "均值回归": self.mean_reversion,
            "动量策略": self.momentum_strategy,
            "突破策略": self.breakout_strategy,
            "网格交易": self.grid_trading,
            "配对交易": self.pairs_trading
        }
        
        # 创建策略智能体
        self.agents = {}
        descriptions = {
            "趋势跟踪": "双均线策略，捕捉趋势性行情",
            "均值回归": "布林带策略，利用价格回归特性",
            "动量策略": "RSI+MACD组合，捕捉动量信号",
            "突破策略": "通道突破，捕捉爆发性行情",
            "网格交易": "震荡市场中的网格套利",
            "配对交易": "统计套利，低风险稳定收益"
        }
        
        for name, func in self.strategy_functions.items():
            agent = StrategyAgent(name, func, descriptions[name])
            self.agents[name] = agent
        
        # 初始化竞赛引擎
        self.contest_engine = ContestEngine()
        for agent in self.agents.values():
            self.contest_engine.register_agent(agent)
    
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

# 回测引擎（保持原有功能）
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

# AI对话处理器（集成竞赛功能）
class QuantGPTContestProcessor:
    def __init__(self):
        self.strategy_engine = EnhancedStrategyEngine()
        self.backtest_engine = BacktestEngine()
        
    def parse_user_input(self, user_input):
        """解析用户输入"""
        user_input = user_input.lower()
        
        # 检查是否是竞赛模式
        contest_keywords = ["竞赛", "contest", "比赛", "对比", "哪个更好", "最优", "排名"]
        is_contest_mode = any(keyword in user_input for keyword in contest_keywords)
        
        # 提取股票代码
        stock_pattern = r'\b[A-Z]{1,5}\b'
        stocks = re.findall(stock_pattern, user_input.upper())
        
        return {
            'stocks': stocks,
            'is_contest_mode': is_contest_mode,
            'original_input': user_input
        }
    
    def run_strategy_contest(self, stock):
        """运行策略竞赛"""
        try:
            # 获取数据
            data = yf.Ticker(stock).history(period="2y")
            if data.empty:
                return None, f"无法获取 {stock} 的数据"
            
            # 添加技术指标
            data = self.add_technical_indicators(data)
            
            # 为所有策略运行回测
            results = {}
            for agent_name, agent in self.strategy_engine.agents.items():
                try:
                    # 获取默认参数
                    params = self.get_default_params(agent_name)
                    
                    # 运行策略
                    strategy_data, description = agent.execute_strategy(data, params)[:2]
                    
                    # 运行回测
                    backtest_data = self.backtest_engine.run_backtest(strategy_data)
                    
                    # 计算指标
                    metrics = self.backtest_engine.calculate_metrics(backtest_data)
                    
                    results[agent_name] = {
                        'backtest_data': backtest_data,
                        'metrics': metrics,
                        'description': description,
                        'params': params
                    }
                    
                except Exception as e:
                    results[agent_name] = {
                        'error': str(e)
                    }
            
            # 运行竞赛
            contest_result = self.strategy_engine.contest_engine.run_contest(results)
            
            return results, contest_result
            
        except Exception as e:
            return None, f"竞赛运行失败：{str(e)}"
    
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
    
    def generate_contest_response(self, stock, results, contest_result):
        """生成竞赛结果响应"""
        if results is None:
            return f"❌ {contest_result}"
        
        # 获取策略排名
        rankings = self.strategy_engine.contest_engine.get_strategy_rankings(contest_result)
        
        response = f"## 🏆 {stock} 策略竞赛结果\n\n"
        response += "**竞赛机制：** 基于ContestTrade框架，通过"量化-预测-分配"三阶段评估\n\n"
        response += "### 📊 策略排名 (按综合得分)\n\n"
        
        for i, ranking in enumerate(rankings):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            response += f"{medal} **{ranking['name']}** (权重: {ranking['weight']:.1%})\n"
            response += f"   • 综合得分: {ranking['score']:.3f}\n"
            response += f"   • 夏普比率: {ranking['sharpe']:.2f}\n"
            response += f"   • 胜率: {ranking['win_rate']:.1%}\n"
            response += f"   • 策略说明: {ranking['description']}\n\n"
        
        # 推荐组合
        winner = rankings[0]
        response += "### 🎯 AI推荐\n"
        response += f"**冠军策略：** {winner['name']}\n"
        response += f"**推荐理由：** 综合得分最高({winner['score']:.3f})，建议分配{winner['weight']:.1%}的资金权重\n\n"
        
        # 市场适应性分析
        top3_strategies = [r['name'] for r in rankings[:3]]
        response += "### 🧠 智能洞察\n"
        response += f"**当前市场特征：** 前三名策略为 {', '.join(top3_strategies)}\n"
        response += f"**建议操作：** 采用动态权重分配，重点关注{winner['name']}策略信号\n"
        
        return response

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

# 竞赛结果可视化
def show_contest_chart(stock, results, contest_result):
    """显示竞赛结果图表"""
    if not results:
        st.error("没有竞赛数据可显示")
        return
    
    # 获取排名
    rankings = st.session_state.processor.strategy_engine.contest_engine.get_strategy_rankings(contest_result)
    
    # 创建权重分布饼图
    col1, col2 = st.columns(2)
    
    with col1:
        weights_df = pd.DataFrame(rankings)
        fig_pie = px.pie(weights_df, values='weight', names='name', 
                        title=f'{stock} 策略权重分配',
                        color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 策略表现对比
        scores_df = weights_df.copy()
        fig_bar = px.bar(scores_df, x='name', y='score', 
                        title='策略综合得分对比',
                        color='score',
                        color_continuous_scale='viridis')
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # 显示冠军策略的详细图表
    winner_name = rankings[0]['name']
    if winner_name in results and 'backtest_data' in results[winner_name]:
        st.subheader(f"🏆 冠军策略详细分析: {winner_name}")
        
        winner_data = results[winner_name]['backtest_data']
        
        # 创建详细的策略分析图表
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{stock} - {winner_name} 价格走势与信号', '策略收益对比', '权重分配历史'),
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # 价格和信号
        fig.add_trace(
            go.Scatter(x=winner_data.index, y=winner_data['Close'], 
                      name='收盘价', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 买入信号
        buy_signals = winner_data[winner_data['Position'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                          mode='markers', name='买入信号', 
                          marker=dict(color='green', size=8, symbol='triangle-up')),
                row=1, col=1
            )
        
        # 卖出信号
        sell_signals = winner_data[winner_data['Position'] == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                          mode='markers', name='卖出信号',
                          marker=dict(color='red', size=8, symbol='triangle-down')),
                row=1, col=1
            )
        
        # 策略收益对比
        fig.add_trace(
            go.Scatter(x=winner_data.index, y=(winner_data['Cumulative_Returns']-1)*100,
                      name='买入持有收益(%)', line=dict(color='gray')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=winner_data.index, y=(winner_data['Strategy_Cumulative']-1)*100,
                      name=f'{winner_name}策略收益(%)', line=dict(color='gold', width=3)),
            row=2, col=1
        )
        
        # 权重历史（模拟数据）
        weight_history = [rankings[0]['weight']] * len(winner_data)
        fig.add_trace(
            go.Scatter(x=winner_data.index, y=weight_history,
                      name='策略权重', line=dict(color='purple')),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=True, title=f"🏆 {winner_name} 冠军策略完整分析")
        st.plotly_chart(fig, use_container_width=True)

# 主应用
def main():
    st.title("🏆 QuantGPT Contest - AI竞赛量化交易助手")
    st.markdown("**基于ContestTrade框架的多策略智能竞赛平台**")
    st.markdown("---")
    
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """🏆 欢迎使用QuantGPT Contest！我是您的AI竞赛量化交易助手。

**新功能亮点：**
🔥 **策略竞赛模式** - 基于ContestTrade框架
🔥 **智能权重分配** - 动态评估策略表现
🔥 **表现预测** - AI预测策略未来表现

**可用功能：**
• **单策略分析：** "分析AAPL的趋势策略"
• **策略竞赛：** "AAPL策略竞赛" 或 "哪个策略最适合TSLA"
• **对比分析：** "比较GOOGL的所有策略表现"

**竞赛机制说明：**
1. **量化阶段** - 评估所有策略的历史表现
2. **预测阶段** - AI预测各策略未来表现
3. **分配阶段** - 智能分配资金权重

请告诉我您想分析什么？例如：
'AAPL策略竞赛'
'比较TSLA的所有策略'
'MSFT哪个策略最好'"""}
        ]
    
    if "processor" not in st.session_state:
        st.session_state.processor = QuantGPTContestProcessor()
    
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
        with st.spinner("🤖 正在运行策略竞赛..."):
            parsed_input = st.session_state.processor.parse_user_input(user_input)
            
            if not parsed_input['stocks']:
                response = "🤖 请告诉我您想分析的股票代码，例如：'AAPL策略竞赛' 或 'TSLA哪个策略最好？'"
            else:
                stock = parsed_input['stocks'][0]  # 取第一个股票
                
                if parsed_input['is_contest_mode']:
                    # 运行策略竞赛
                    results, contest_result = st.session_state.processor.run_strategy_contest(stock)
                    response = st.session_state.processor.generate_contest_response(stock, results, contest_result)
                    
                    # 存储竞赛结果用于可视化
                    if 'contest_results' not in st.session_state:
                        st.session_state.contest_results = {}
                    st.session_state.contest_results[stock] = {
                        'results': results,
                        'contest_result': contest_result
                    }
                else:
                    response = "🤖 我检测到您想要单策略分析。要不要试试策略竞赛模式？输入 'AAPL策略竞赛' 来对比所有策略的表现！"
        
        # 添加AI响应
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_message(response, False)
        
        # 如果有竞赛结果，显示图表选项
        if 'contest_results' in st.session_state and st.session_state.contest_results:
            st.markdown("---")
            st.subheader("📊 竞赛结果可视化")
            
            cols = st.columns(len(st.session_state.contest_results))
            for i, stock in enumerate(st.session_state.contest_results.keys()):
                with cols[i]:
                    if st.button(f"🏆 {stock} 竞赛图表", key=f"contest_chart_{stock}"):
                        contest_data = st.session_state.contest_results[stock]
                        show_contest_chart(stock, contest_data['results'], contest_data['contest_result'])
        
        # 重新运行显示更新
        st.rerun()
    
    # 侧边栏功能
    with st.sidebar:
        st.title("🏆 竞赛控制台")
        
        # 竞赛统计
        if hasattr(st.session_state, 'processor'):
            contest_engine = st.session_state.processor.strategy_engine.contest_engine
            st.subheader("📈 竞赛统计")
            st.metric("历史竞赛次数", len(contest_engine.contest_history))
            st.metric("注册策略数量", len(contest_engine.agents))
        
        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = [st.session_state.messages[0]]  # 保留欢迎消息
            if 'contest_results' in st.session_state:
                del st.session_state.contest_results
            st.rerun()
        
        if st.button("📋 竞赛示例"):
            examples = [
                "AAPL策略竞赛",
                "TSLA哪个策略最好？",
                "比较GOOGL所有策略表现",
                "MSFT策略对比分析",
                "NVDA最优策略组合"
            ]
            st.write("💡 **竞赛示例问题：**")
            for example in examples:
                st.write(f"• {example}")
        
        st.markdown("---")
        st.markdown("### 🔬 ContestTrade框架")
        st.markdown("**三阶段竞赛机制：**")
        st.markdown("1. **量化** - 评估历史表现")
        st.markdown("2. **预测** - AI预测未来表现") 
        st.markdown("3. **分配** - 智能权重分配")
        
        st.markdown("---")
        st.markdown("**参与竞赛的策略：**")
        st.markdown("🥇 趋势跟踪（双均线）")
        st.markdown("🥈 均值回归（布林带）") 
        st.markdown("🥉 动量策略（RSI+MACD）")
        st.markdown("🏅 突破策略（通道突破）")
        st.markdown("🏅 网格交易")
        st.markdown("🏅 配对交易")
        
        st.markdown("---")
        st.markdown("### 🎯 竞赛优势")
        st.markdown("✅ **动态权重分配**")
        st.markdown("✅ **AI表现预测**")
        st.markdown("✅ **市场适应性**")
        st.markdown("✅ **策略组合优化**")
        
        # 高级设置
        with st.expander("⚙️ 高级设置"):
            st.slider("历史表现权重", 0.0, 1.0, 0.7, 0.1, key="history_weight")
            st.slider("预测表现权重", 0.0, 1.0, 0.3, 0.1, key="prediction_weight")
            st.selectbox("市场状态", ["Normal", "Volatile", "Trending", "Ranging"], key="market_regime")

if __name__ == "__main__":
    main()
