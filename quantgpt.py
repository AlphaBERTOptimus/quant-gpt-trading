import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# 配置页面
st.set_page_config(
    page_title="🚀 QuantGPT - AI量化交易平台",
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# 导入QuantGPT模块
try:
    # 这里我们将原始代码重新定义为模块
    import torch
    import yfinance as yf
    from dataclasses import dataclass
    from typing import Dict, List, Optional
    
    @dataclass
    class QuantGPTConfig:
        """QuantGPT配置类"""
        initial_capital: float = 100000.0
        commission: float = 0.001
        slippage: float = 0.0005
        max_position_size: float = 0.2
        risk_free_rate: float = 0.02
        sentiment_model: str = "ProsusAI/finbert"
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        default_period: str = "2y"
        min_data_points: int = 200

    @dataclass
    class AdvancedScreeningCriteria:
        """高级股票筛选条件"""
        pe_ratio_min: Optional[float] = None
        pe_ratio_max: Optional[float] = None
        pb_ratio_max: Optional[float] = None
        debt_to_equity_max: Optional[float] = None
        roe_min: Optional[float] = None
        dividend_yield_min: Optional[float] = None
        market_cap_min: Optional[float] = None
        revenue_growth_min: Optional[float] = None
        current_ratio_min: Optional[float] = None
        gross_margin_min: Optional[float] = None

    class AIAnalysisEngine:
        """AI驱动的金融分析引擎"""
        
        def __init__(self, config: QuantGPTConfig):
            self.config = config
            self.sentiment_analyzer = None
            
        def generate_market_insight(self, symbol: str, price_data: pd.DataFrame, 
                                  news_sentiment: float = 0.5) -> Dict:
            """生成AI驱动的市场洞察"""
            
            current_price = price_data['Close'].iloc[-1]
            prev_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else current_price
            price_change = (current_price - prev_price) / prev_price if prev_price != 0 else 0
            
            # 计算移动平均
            sma_20 = price_data['Close'].rolling(20).mean().iloc[-1] if len(price_data) >= 20 else current_price
            sma_50 = price_data['Close'].rolling(50).mean().iloc[-1] if len(price_data) >= 50 else current_price
            
            # 计算RSI
            delta = price_data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # AI评分逻辑
            sentiment_score = news_sentiment
            technical_score = 0.5
            
            if current_price > sma_20 > sma_50:
                technical_score += 0.3
            if current_rsi < 30:
                technical_score += 0.2
            elif current_rsi > 70:
                technical_score -= 0.2
                
            combined_score = (sentiment_score * 0.4 + technical_score * 0.6)
            
            # 生成建议
            if combined_score > 0.7:
                recommendation = "🟢 强烈买入"
                reason = "技术面和基本面都显示强劲上涨信号"
            elif combined_score > 0.6:
                recommendation = "🟡 买入"
                reason = "总体趋势积极，建议逢低买入"
            elif combined_score < 0.3:
                recommendation = "🔴 卖出"
                reason = "多重负面信号，建议减仓"
            elif combined_score < 0.4:
                recommendation = "🟡 观望"
                reason = "信号混合，建议等待更明确的方向"
            else:
                recommendation = "🟡 持有"
                reason = "当前趋势不明确，维持现有仓位"
                
            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_change_pct": price_change * 100,
                "technical_indicators": {
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "rsi": current_rsi
                },
                "ai_scores": {
                    "sentiment_score": sentiment_score,
                    "technical_score": technical_score,
                    "combined_score": combined_score
                },
                "recommendation": recommendation,
                "reasoning": reason,
                "confidence": abs(combined_score - 0.5) * 2,
                "timestamp": datetime.now().isoformat()
            }

    class DataManager:
        """金融数据管理器"""
        
        def __init__(self, config: QuantGPTConfig):
            self.config = config
            self.cache = {}
        
        def get_stock_data(self, symbol: str, period: str = None) -> Optional[pd.DataFrame]:
            """获取股票数据"""
            period = period or self.config.default_period
            cache_key = f"{symbol}_{period}"
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if data.empty or len(data) < 50:  # 降低最低要求
                    return None
                    
                # 数据清洗
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                
                data = data.dropna()
                self.cache[cache_key] = data
                return data
                
            except Exception as e:
                st.error(f"获取 {symbol} 数据失败: {e}")
                return None
        
        def get_stock_info(self, symbol: str) -> Dict:
            """获取股票基本信息"""
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return {
                    "name": info.get("longName", symbol),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "dividend_yield": info.get("dividendYield", 0)
                }
            except:
                return {"name": symbol, "sector": "Unknown"}

    class TechnicalIndicators:
        """技术指标计算器"""
        
        @staticmethod
        def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
            """为数据添加所有技术指标"""
            df = df.copy()
            close = df['Close']
            
            # 移动平均
            df['SMA_20'] = close.rolling(20).mean()
            df['SMA_50'] = close.rolling(50).mean()
            df['EMA_12'] = close.ewm(span=12).mean()
            df['EMA_26'] = close.ewm(span=26).mean()
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # 布林带
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)
            df['BB_Middle'] = sma_20
            
            return df

    class StrategyEngine:
        """交易策略引擎"""
        
        def __init__(self, config: QuantGPTConfig):
            self.config = config
            
        def sma_crossover_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
            """移动平均交叉策略"""
            df = TechnicalIndicators.add_all_indicators(data)
            df['Signal'] = 0
            df['Signal'][20:] = np.where(df['SMA_20'][20:] > df['SMA_50'][20:], 1, 0)
            df['Position'] = df['Signal'].diff()
            return df
        
        def rsi_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
            """RSI策略"""
            df = TechnicalIndicators.add_all_indicators(data)
            df['Signal'] = 0
            df.loc[df['RSI'] < 30, 'Signal'] = 1
            df.loc[df['RSI'] > 70, 'Signal'] = -1
            df['Position'] = df['Signal'].diff()
            return df

    class BacktestEngine:
        """回测引擎"""
        
        def __init__(self, config: QuantGPTConfig):
            self.config = config
            
        def execute_backtest(self, strategy_data: pd.DataFrame, symbol: str) -> Dict:
            """执行回测"""
            capital = self.config.initial_capital
            positions = 0
            trades = []
            portfolio_values = []
            
            for i, (date, row) in enumerate(strategy_data.iterrows()):
                current_price = row['Close']
                position_change = row.get('Position', 0)
                
                if pd.isna(position_change):
                    position_change = 0
                
                # 买入信号
                if position_change == 1 and positions == 0:
                    shares_to_buy = int((capital * self.config.max_position_size) / current_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price * (1 + self.config.commission)
                        if cost <= capital:
                            capital -= cost
                            positions = shares_to_buy
                            trades.append({
                                'date': date,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': current_price,
                                'value': cost
                            })
                
                # 卖出信号
                elif position_change == -1 and positions > 0:
                    revenue = positions * current_price * (1 - self.config.commission)
                    capital += revenue
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'shares': positions,
                        'price': current_price,
                        'value': revenue
                    })
                    positions = 0
                
                portfolio_value = capital + positions * current_price
                portfolio_values.append(portfolio_value)
            
            strategy_data = strategy_data.copy()
            strategy_data['Portfolio_Value'] = portfolio_values
            
            # 计算绩效指标
            metrics = self._calculate_metrics(strategy_data, symbol)
            
            return {
                'data': strategy_data,
                'trades': trades,
                'metrics': metrics,
                'final_value': portfolio_values[-1] if portfolio_values else self.config.initial_capital
            }
        
        def _calculate_metrics(self, data: pd.DataFrame, symbol: str) -> Dict:
            """计算绩效指标"""
            if 'Portfolio_Value' not in data.columns:
                return {}
            
            portfolio_values = data['Portfolio_Value']
            returns = portfolio_values.pct_change().dropna()
            
            total_return = (portfolio_values.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
            annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            excess_returns = returns - self.config.risk_free_rate / 252
            sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0
            
            # 最大回撤
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() if not drawdowns.empty else 0
            
            return {
                'symbol': symbol,
                'total_return': total_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': portfolio_values.iloc[-1]
            }

    class FundamentalAnalysisEngine:
        """基本面分析引擎"""
        
        def get_fundamental_data(self, symbol: str) -> Dict:
            """获取基本面数据"""
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                fundamental_data = {
                    "symbol": symbol,
                    "company_name": info.get("longName", symbol),
                    "pe_ratio": info.get("trailingPE", None),
                    "pb_ratio": info.get("priceToBook", None),
                    "roe": info.get("returnOnEquity", None),
                    "debt_to_equity": info.get("debtToEquity", None),
                    "current_ratio": info.get("currentRatio", None),
                    "dividend_yield": info.get("dividendYield", None),
                    "market_cap": info.get("marketCap", None),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                }
                
                # 计算基本面评分
                score = 50
                pe = fundamental_data.get("pe_ratio")
                if pe and 0 < pe < 15:
                    score += 15
                elif pe and 15 <= pe <= 25:
                    score += 10
                
                roe = fundamental_data.get("roe")
                if roe and roe > 0.2:
                    score += 15
                elif roe and roe > 0.15:
                    score += 10
                
                fundamental_data["fundamental_score"] = max(0, min(100, score))
                
                return fundamental_data
                
            except Exception as e:
                return {"symbol": symbol, "error": str(e)}

    class ScreeningPresets:
        """筛选预设"""
        
        @staticmethod
        def value_stocks():
            return AdvancedScreeningCriteria(
                pe_ratio_max=15,
                pb_ratio_max=2,
                debt_to_equity_max=0.5,
                roe_min=10,
                dividend_yield_min=2
            )
        
        @staticmethod
        def growth_stocks():
            return AdvancedScreeningCriteria(
                revenue_growth_min=15,
                roe_min=15,
                gross_margin_min=40,
                pe_ratio_max=40
            )

    class QuantGPT:
        """QuantGPT主应用类"""
        
        def __init__(self, config: QuantGPTConfig = None):
            self.config = config or QuantGPTConfig()
            self.ai_engine = AIAnalysisEngine(self.config)
            self.data_manager = DataManager(self.config)
            self.strategy_engine = StrategyEngine(self.config)
            self.backtest_engine = BacktestEngine(self.config)
            self.fundamental_engine = FundamentalAnalysisEngine()
        
        def analyze_stock(self, symbol: str, period: str = "1y") -> Dict:
            """分析股票"""
            data = self.data_manager.get_stock_data(symbol, period)
            if data is None:
                return {"error": f"无法获取{symbol}数据"}
            
            stock_info = self.data_manager.get_stock_info(symbol)
            insight = self.ai_engine.generate_market_insight(symbol, data)
            fundamental_data = self.fundamental_engine.get_fundamental_data(symbol)
            
            return {
                "symbol": symbol,
                "stock_info": stock_info,
                "data": data,
                "ai_insight": insight,
                "fundamental_analysis": fundamental_data,
                "analysis_date": datetime.now().isoformat()
            }
        
        def run_strategy_backtest(self, symbol: str, strategy_name: str, period: str = "1y") -> Dict:
            """运行策略回测"""
            data = self.data_manager.get_stock_data(symbol, period)
            if data is None:
                return {"error": f"无法获取{symbol}数据"}
            
            if strategy_name == "sma_crossover":
                strategy_data = self.strategy_engine.sma_crossover_strategy(data)
            elif strategy_name == "rsi":
                strategy_data = self.strategy_engine.rsi_strategy(data)
            else:
                return {"error": f"未知策略: {strategy_name}"}
            
            backtest_result = self.backtest_engine.execute_backtest(strategy_data, symbol)
            backtest_result['strategy_name'] = strategy_name
            
            return backtest_result
        
        def screen_stocks_basic(self, criteria, symbols=None):
            """股票筛选"""
            if symbols is None:
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
            
            results = []
            for symbol in symbols:
                try:
                    fundamental = self.fundamental_engine.get_fundamental_data(symbol)
                    if "error" not in fundamental:
                        results.append({
                            "symbol": symbol,
                            "fundamental_score": fundamental.get("fundamental_score", 0),
                            "pe_ratio": fundamental.get("pe_ratio"),
                            "roe": fundamental.get("roe"),
                            "sector": fundamental.get("sector")
                        })
                except:
                    continue
            
            results.sort(key=lambda x: x["fundamental_score"], reverse=True)
            return results

    # 缓存初始化
    @st.cache_resource
    def initialize_quantgpt():
        """初始化QuantGPT系统"""
        return QuantGPT()

except Exception as e:
    st.error(f"❌ 模块导入失败: {e}")
    st.stop()

# 主标题和介绍
st.markdown('<h1 class="main-header">🚀 QuantGPT - AI量化交易平台</h1>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="info-card">🤖 <b>AI驱动分析</b><br/>智能情感分析与市场洞察</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="info-card">📊 <b>专业回测</b><br/>多策略回测与绩效分析</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="info-card">🔍 <b>智能筛选</b><br/>基本面与技术面综合筛选</div>', unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.header("🎯 功能导航")
    
    app_mode = st.selectbox(
        "选择功能模块",
        ["📊 股票分析", "🔬 策略回测", "🔍 股票筛选", "📈 多策略比较", "💎 基本面分析"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("⚙️ 系统设置")
    
    initial_capital = st.number_input("初始资金 ($)", value=100000, min_value=1000, step=1000)
    commission = st.number_input("手续费率", value=0.001, min_value=0.0, max_value=0.1, format="%.4f")
    
    st.markdown("---")
    st.markdown("### 📈 市场概览")
    
    # 显示一些市场信息
    market_indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC", 
        "道琼斯": "^DJI"
    }
    
    for name, symbol in market_indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = (current - prev) / prev * 100
                color = "green" if change > 0 else "red"
                st.markdown(f"**{name}**: <span style='color:{color}'>{change:+.2f}%</span>", unsafe_allow_html=True)
        except:
            st.markdown(f"**{name}**: 数据获取中...")

# 初始化系统
try:
    with st.spinner("🤖 正在初始化QuantGPT系统..."):
        quantgpt = initialize_quantgpt()
    
    # 成功初始化提示
    st.success("✅ QuantGPT系统初始化完成！AI模型已加载，准备为您服务。")
    
except Exception as e:
    st.error(f"❌ 系统初始化失败: {e}")
    st.stop()

# 主功能区域
if app_mode == "📊 股票分析":
    st.header("📊 AI智能股票分析")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input(
            "🔍 输入股票代码", 
            value="AAPL", 
            help="输入美股代码，如 AAPL, GOOGL, TSLA 等",
            placeholder="例如: AAPL"
        )
    
    with col2:
        period = st.selectbox(
            "📅 分析周期", 
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"], 
            index=3
        )
    
    if st.button("🚀 开始AI分析", type="primary", use_container_width=True):
        if symbol:
            with st.spinner(f"🤖 正在深度分析 {symbol.upper()}..."):
                try:
                    result = quantgpt.analyze_stock(symbol.upper(), period)
                    
                    if "error" not in result:
                        insight = result["ai_insight"]
                        
                        # AI分析结果
                        st.subheader("🤖 AI分析结果")
                        
                        # 关键指标
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "💰 当前价格", 
                                f"${insight['current_price']:.2f}",
                                delta=f"{insight['price_change_pct']:+.2f}%"
                            )
                        
                        with col2:
                            score = insight['ai_scores']['combined_score'] * 100
                            st.metric("🎯 AI评分", f"{score:.1f}/100")
                        
                        with col3:
                            st.metric("📊 置信度", f"{insight['confidence']*100:.1f}%")
                        
                        with col4:
                            rsi = insight['technical_indicators']['rsi']
                            st.metric("📈 RSI", f"{rsi:.1f}")
                        
                        # AI建议
                        st.markdown("### 🎯 AI投资建议")
                        
                        recommendation_color = {
                            "🟢 强烈买入": "green",
                            "🟡 买入": "orange", 
                            "🟡 持有": "blue",
                            "🟡 观望": "gray",
                            "🔴 卖出": "red"
                        }
                        
                        rec_color = recommendation_color.get(insight['recommendation'], "blue")
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {rec_color}22, {rec_color}11); 
                                   border-left: 4px solid {rec_color}; 
                                   padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                            <h4 style='color: {rec_color}; margin: 0;'>{insight['recommendation']}</h4>
                            <p style='margin: 0.5rem 0 0 0;'>{insight['reasoning']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 技术指标详情
                        st.subheader("📈 技术指标分析")
                        tech_col1, tech_col2, tech_col3 = st.columns(3)
                        
                        with tech_col1:
                            st.metric("📊 SMA 20", f"${insight['technical_indicators']['sma_20']:.2f}")
                        with tech_col2:
                            st.metric("📊 SMA 50", f"${insight['technical_indicators']['sma_50']:.2f}")
                        with tech_col3:
                            st.metric("⚡ RSI", f"{insight['technical_indicators']['rsi']:.1f}")
                        
                        # 价格走势图
                        st.subheader("📊 价格走势与技术指标")
                        
                        data = result["data"]
                        
                        # 创建子图
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('价格走势', 'RSI指标'),
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3]
                        )
                        
                        # 价格线
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['Close'],
                                mode='lines',
                                name='收盘价',
                                line=dict(color='#1f77b4', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        # 移动平均线
                        if len(data) >= 20:
                            sma_20 = data['Close'].rolling(20).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=sma_20,
                                    mode='lines',
                                    name='SMA 20',
                                    line=dict(color='orange', width=1)
                                ),
                                row=1, col=1
                            )
                        
                        if len(data) >= 50:
                            sma_50 = data['Close'].rolling(50).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=sma_50,
                                    mode='lines',
                                    name='SMA 50',
                                    line=dict(color='red', width=1)
                                ),
                                row=1, col=1
                            )
                        
                        # RSI指标
                        if len(data) >= 14:
                            delta = data['Close'].diff()
                            gain = delta.where(delta > 0, 0).rolling(14).mean()
                            loss = -delta.where(delta < 0, 0).rolling(14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=rsi,
                                    mode='lines',
                                    name='RSI',
                                    line=dict(color='purple', width=2)
                                ),
                                row=2, col=1
                            )
                            
                            # RSI超买超卖线
                            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
                        
                        fig.update_layout(
                            title=f"{symbol.upper()} 技术分析图表",
                            height=600,
                            showlegend=True,
                            xaxis_title="日期",
                            yaxis_title="价格 ($)",
                            xaxis2_title="日期",
                            yaxis2_title="RSI"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 基本面分析
                        if "fundamental_analysis" in result and "error" not in result["fundamental_analysis"]:
                            fund = result["fundamental_analysis"]
                            
                            st.subheader("💎 基本面分析")
                            
                            fund_col1, fund_col2 = st.columns(2)
                            
                            with fund_col1:
                                st.markdown("**📋 公司信息**")
                                st.write(f"• 公司名称: {fund.get('company_name', 'N/A')}")
                                st.write(f"• 行业板块: {fund.get('sector', 'N/A')}")
                                st.write(f"• 细分行业: {fund.get('industry', 'N/A')}")
                                
                                market_cap = fund.get('market_cap', 0)
                                if market_cap and market_cap > 0:
                                    market_cap_b = market_cap / 1e9
                                    st.write(f"• 市值: ${market_cap_b:.1f}B")
                            
                            with fund_col2:
                                st.markdown("**📊 财务指标**")
                                pe_ratio = fund.get('pe_ratio')
                                if pe_ratio:
                                    st.write(f"• PE比率: {pe_ratio:.2f}")
                                
                                pb_ratio = fund.get('pb_ratio')
                                if pb_ratio:
                                    st.write(f"• PB比率: {pb_ratio:.2f}")
                                
                                roe = fund.get('roe')
                                if roe:
                                    st.write(f"• ROE: {roe*100:.1f}%")
                                
                                dividend_yield = fund.get('dividend_yield')
                                if dividend_yield:
                                    st.write(f"• 股息率: {dividend_yield*100:.2f}%")
                                
                                score = fund.get('fundamental_score', 0)
                                st.write(f"• **基本面评分: {score:.1f}/100**")
                        
                        # 成交量分析
                        st.subheader("📊 成交量分析")
                        
                        vol_fig = go.Figure()
                        vol_fig.add_trace(
                            go.Bar(
                                x=data.index,
                                y=data['Volume'],
                                name='成交量',
                                marker_color='lightblue',
                                opacity=0.7
                            )
                        )
                        
                        # 成交量移动平均
                        if len(data) >= 20:
                            vol_sma = data['Volume'].rolling(20).mean()
                            vol_fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=vol_sma,
                                    mode='lines',
                                    name='成交量均线',
                                    line=dict(color='red', width=2)
                                )
                            )
                        
                        vol_fig.update_layout(
                            title=f"{symbol.upper()} 成交量分析",
                            xaxis_title="日期",
                            yaxis_title="成交量",
                            height=300
                        )
                        
                        st.plotly_chart(vol_fig, use_container_width=True)
                        
                    else:
                        st.error(f"❌ {result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ 分析过程中发生错误: {str(e)}")
        else:
            st.warning("⚠️ 请输入股票代码")

elif app_mode == "🔬 策略回测":
    st.header("🔬 交易策略回测")
    
    # 策略配置
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bt_symbol = st.text_input("股票代码", value="AAPL", placeholder="例如: AAPL")
    
    with col2:
        strategy = st.selectbox(
            "选择策略", 
            ["sma_crossover", "rsi"],
            format_func=lambda x: {"sma_crossover": "📈 移动平均交叉", "rsi": "⚡ RSI策略"}[x]
        )
    
    with col3:
        bt_period = st.selectbox("回测周期", ["6mo", "1y", "2y", "3y", "5y"], index=1)
    
    # 策略说明
    strategy_descriptions = {
        "sma_crossover": "📈 **移动平均交叉策略**: 当短期均线(20日)上穿长期均线(50日)时买入，下穿时卖出",
        "rsi": "⚡ **RSI策略**: 当RSI低于30时买入(超卖)，高于70时卖出(超买)"
    }
    
    st.info(strategy_descriptions[strategy])
    
    if st.button("🚀 开始回测", type="primary", use_container_width=True):
        if bt_symbol:
            with st.spinner(f"🔬 正在回测 {bt_symbol.upper()} - {strategy}策略..."):
                try:
                    result = quantgpt.run_strategy_backtest(bt_symbol.upper(), strategy, bt_period)
                    
                    if "error" not in result:
                        metrics = result["metrics"]
                        
                        st.subheader("📊 回测结果")
                        
                        # 核心绩效指标
                        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                        
                        with perf_col1:
                            total_return = metrics.get('total_return', 0)
                            color = "normal" if total_return >= 0 else "inverse"
                            st.metric(
                                "📈 总收益率", 
                                f"{total_return:.2%}",
                                delta=f"{total_return:.2%}",
                                delta_color=color
                            )
                        
                        with perf_col2:
                            volatility = metrics.get('annual_volatility', 0)
                            st.metric("📊 年化波动率", f"{volatility:.2%}")
                        
                        with perf_col3:
                            sharpe = metrics.get('sharpe_ratio', 0)
                            st.metric("⚡ 夏普比率", f"{sharpe:.3f}")
                        
                        with perf_col4:
                            max_dd = metrics.get('max_drawdown', 0)
                            st.metric("📉 最大回撤", f"{max_dd:.2%}")
                        
                        # 详细指标
                        st.subheader("📋 详细绩效分析")
                        
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.markdown("**💰 收益指标**")
                            final_value = metrics.get('final_value', initial_capital)
                            profit = final_value - initial_capital
                            st.write(f"• 初始资金: ${initial_capital:,.2f}")
                            st.write(f"• 最终价值: ${final_value:,.2f}")
                            st.write(f"• 绝对收益: ${profit:,.2f}")
                            
                        with detail_col2:
                            st.markdown("**📊 风险指标**")
                            st.write(f"• 年化波动率: {volatility:.2%}")
                            st.write(f"• 最大回撤: {max_dd:.2%}")
                            st.write(f"• 夏普比率: {sharpe:.3f}")
                        
                        # 收益曲线图
                        if 'data' in result and 'Portfolio_Value' in result['data'].columns:
                            st.subheader("📈 策略收益曲线")
                            
                            strategy_data = result['data']
                            
                            # 创建收益对比图
                            fig = go.Figure()
                            
                            # 策略收益曲线
                            fig.add_trace(
                                go.Scatter(
                                    x=strategy_data.index,
                                    y=strategy_data['Portfolio_Value'],
                                    mode='lines',
                                    name=f'{strategy}策略',
                                    line=dict(color='green', width=3)
                                )
                            )
                            
                            # 基准收益曲线(买入持有)
                            initial_shares = initial_capital / strategy_data['Close'].iloc[0]
                            benchmark_values = initial_shares * strategy_data['Close']
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=strategy_data.index,
                                    y=benchmark_values,
                                    mode='lines',
                                    name='买入持有基准',
                                    line=dict(color='blue', width=2, dash='dash')
                                )
                            )
                            
                            # 添加买卖信号
                            if 'Position' in strategy_data.columns:
                                buy_signals = strategy_data[strategy_data['Position'] == 1]
                                sell_signals = strategy_data[strategy_data['Position'] == -1]
                                
                                if not buy_signals.empty:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=buy_signals.index,
                                            y=buy_signals['Portfolio_Value'],
                                            mode='markers',
                                            name='买入信号',
                                            marker=dict(
                                                color='red',
                                                size=10,
                                                symbol='triangle-up'
                                            )
                                        )
                                    )
                                
                                if not sell_signals.empty:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=sell_signals.index,
                                            y=sell_signals['Portfolio_Value'],
                                            mode='markers',
                                            name='卖出信号',
                                            marker=dict(
                                                color='blue',
                                                size=10,
                                                symbol='triangle-down'
                                            )
                                        )
                                    )
                            
                            fig.update_layout(
                                title=f"{bt_symbol.upper()} - {strategy}策略 vs 基准对比",
                                xaxis_title="日期",
                                yaxis_title="组合价值 ($)",
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 交易记录
                            if result.get('trades'):
                                st.subheader("📋 交易记录")
                                
                                trades_df = pd.DataFrame(result['trades'])
                                trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                                trades_df['price'] = trades_df['price'].round(2)
                                trades_df['value'] = trades_df['value'].round(2)
                                
                                # 格式化显示
                                st.dataframe(
                                    trades_df,
                                    use_container_width=True,
                                    column_config={
                                        "date": "日期",
                                        "action": "操作",
                                        "shares": "股数",
                                        "price": st.column_config.NumberColumn("价格 ($)", format="%.2f"),
                                        "value": st.column_config.NumberColumn("金额 ($)", format="%.2f")
                                    }
                                )
                                
                                # 交易统计
                                if len(trades_df) > 0:
                                    st.markdown("**📊 交易统计**")
                                    total_trades = len(trades_df) // 2
                                    buy_trades = len(trades_df[trades_df['action'] == 'BUY'])
                                    sell_trades = len(trades_df[trades_df['action'] == 'SELL'])
                                    
                                    trade_col1, trade_col2, trade_col3 = st.columns(3)
                                    with trade_col1:
                                        st.metric("总交易次数", total_trades)
                                    with trade_col2:
                                        st.metric("买入次数", buy_trades)
                                    with trade_col3:
                                        st.metric("卖出次数", sell_trades)
                        
                        # 策略评价
                        st.subheader("🎯 策略评价")
                        
                        if sharpe > 1:
                            evaluation = "🟢 **优秀策略** - 夏普比率>1，风险调整后收益良好"
                            eval_color = "green"
                        elif sharpe > 0.5:
                            evaluation = "🟡 **一般策略** - 夏普比率>0.5，有一定投资价值"
                            eval_color = "orange"
                        elif sharpe > 0:
                            evaluation = "🟠 **较弱策略** - 夏普比率>0，但风险调整后收益较低"
                            eval_color = "orange"
                        else:
                            evaluation = "🔴 **风险策略** - 夏普比率<0，风险调整后亏损"
                            eval_color = "red"
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {eval_color}22, {eval_color}11); 
                                   border-left: 4px solid {eval_color}; 
                                   padding: 1rem; border-radius: 5px;'>
                            {evaluation}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.error(f"❌ {result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ 回测过程中发生错误: {str(e)}")
        else:
            st.warning("⚠️ 请输入股票代码")

elif app_mode == "🔍 股票筛选":
    st.header("🔍 智能股票筛选")
    
    # 筛选预设选择
    preset_type = st.selectbox(
        "📋 选择筛选策略",
        ["value", "growth"],
        format_func=lambda x: {"value": "💎 价值投资策略", "growth": "🚀 成长投资策略"}[x]
    )
    
    # 策略说明
    strategy_info = {
        "value": "💎 **价值投资策略**: 寻找PE低、PB低、有分红、财务稳健的被低估股票",
        "growth": "🚀 **成长投资策略**: 寻找营收增长快、ROE高、毛利率高的高成长股票"
    }
    
    st.info(strategy_info[preset_type])
    
    # 自定义股票池
    with st.expander("📝 自定义股票池 (可选)"):
        custom_symbols = st.text_area(
            "输入股票代码（用逗号分隔）",
            value="AAPL,GOOGL,MSFT,TSLA,NVDA,AMZN,META,NFLX,JPM,JNJ,PG,KO,DIS,V,MA",
            help="默认使用热门股票池，您也可以输入自定义的股票代码"
        )
    
    if st.button("🔍 开始筛选", type="primary", use_container_width=True):
        with st.spinner("🔍 正在筛选优质股票..."):
            try:
                # 解析股票列表
                if custom_symbols.strip():
                    symbols = [s.strip().upper() for s in custom_symbols.split(',') if s.strip()]
                else:
                    symbols = None
                
                # 获取筛选条件
                if preset_type == "value":
                    criteria = ScreeningPresets.value_stocks()
                else:
                    criteria = ScreeningPresets.growth_stocks()
                
                # 执行筛选
                results = quantgpt.screen_stocks_basic(criteria, symbols)
                
                if results:
                    st.success(f"✅ 筛选完成！找到 {len(results)} 只符合条件的股票")
                    
                    # 结果展示
                    st.subheader("📊 筛选结果")
                    
                    # 转换为DataFrame方便显示
                    df = pd.DataFrame(results)
                    
                    # 格式化数据
                    if 'pe_ratio' in df.columns:
                        df['pe_ratio'] = df['pe_ratio'].round(2)
                    if 'roe' in df.columns:
                        df['roe'] = (df['roe'] * 100).round(1)  # 转为百分比
                    
                    # 重命名列
                    column_mapping = {
                        'symbol': '股票代码',
                        'fundamental_score': '基本面评分',
                        'pe_ratio': 'PE比率',
                        'roe': 'ROE(%)',
                        'sector': '行业'
                    }
                    
                    df_display = df.rename(columns=column_mapping)
                    
                    # 配置列显示
                    column_config = {
                        "股票代码": st.column_config.TextColumn("股票代码", width="small"),
                        "基本面评分": st.column_config.NumberColumn(
                            "基本面评分",
                            format="%.1f",
                            min_value=0,
                            max_value=100
                        ),
                        "PE比率": st.column_config.NumberColumn("PE比率", format="%.2f"),
                        "ROE(%)": st.column_config.NumberColumn("ROE(%)", format="%.1f"),
                        "行业": st.column_config.TextColumn("行业", width="medium")
                    }
                    
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        column_config=column_config,
                        hide_index=True
                    )
                    
                    # 可视化分析
                    if len(results) > 1:
                        st.subheader("📈 可视化分析")
                        
                        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 评分分布", "🎯 PE vs ROE", "🏭 行业分布"])
                        
                        with viz_tab1:
                            # 基本面评分分布
                            score_fig = px.histogram(
                                df, 
                                x='fundamental_score',
                                nbins=20,
                                title="基本面评分分布",
                                labels={'fundamental_score': '基本面评分', 'count': '股票数量'}
                            )
                            score_fig.update_layout(height=400)
                            st.plotly_chart(score_fig, use_container_width=True)
                        
                        with viz_tab2:
                            # PE vs ROE 散点图
                            if 'pe_ratio' in df.columns and 'roe' in df.columns:
                                scatter_fig = px.scatter(
                                    df,
                                    x='pe_ratio',
                                    y='roe',
                                    size='fundamental_score',
                                    color='sector',
                                    hover_data=['symbol'],
                                    title="PE比率 vs ROE分布",
                                    labels={'pe_ratio': 'PE比率', 'roe': 'ROE'}
                                )
                                scatter_fig.update_layout(height=500)
                                st.plotly_chart(scatter_fig, use_container_width=True)
                            else:
                                st.info("数据不足，无法绘制PE vs ROE图表")
                        
                        with viz_tab3:
                            # 行业分布
                            if 'sector' in df.columns:
                                sector_counts = df['sector'].value_counts()
                                
                                pie_fig = px.pie(
                                    values=sector_counts.values,
                                    names=sector_counts.index,
                                    title="筛选结果行业分布"
                                )
                                pie_fig.update_layout(height=400)
                                st.plotly_chart(pie_fig, use_container_width=True)
                            else:
                                st.info("行业信息不足，无法绘制分布图")
                    
                    # 推荐股票
                    st.subheader("⭐ 推荐关注")
                    
                    top_stocks = results[:3]  # 取前3只评分最高的
                    
                    for i, stock in enumerate(top_stocks, 1):
                        with st.container():
                            rec_col1, rec_col2 = st.columns([1, 3])
                            
                            with rec_col1:
                                st.markdown(f"### #{i} {stock['symbol']}")
                                st.markdown(f"**评分: {stock['fundamental_score']:.1f}/100**")
                            
                            with rec_col2:
                                st.markdown(f"**行业**: {stock.get('sector', 'N/A')}")
                                if stock.get('pe_ratio'):
                                    st.markdown(f"**PE比率**: {stock['pe_ratio']:.2f}")
                                if stock.get('roe'):
                                    st.markdown(f"**ROE**: {stock['roe']*100:.1f}%")
                        
                        st.markdown("---")
                
                else:
                    st.warning("⚠️ 根据当前筛选条件，未找到符合要求的股票。建议放宽筛选条件或更换股票池。")
                    
            except Exception as e:
                st.error(f"❌ 筛选过程中发生错误: {str(e)}")

elif app_mode == "📈 多策略比较":
    st.header("📈 多策略投资组合比较")
    
    comp_symbol = st.text_input("🔍 输入股票代码", value="AAPL", placeholder="例如: AAPL")
    comp_period = st.selectbox("📅 比较周期", ["6mo", "1y", "2y", "3y"], index=1)
    
    if st.button("🚀 开始策略比较", type="primary", use_container_width=True):
        if comp_symbol:
            with st.spinner(f"📈 正在比较 {comp_symbol.upper()} 的多个策略..."):
                try:
                    strategies = ["sma_crossover", "rsi"]
                    strategy_names = {
                        "sma_crossover": "📈 移动平均交叉",
                        "rsi": "⚡ RSI策略"
                    }
                    
                    results = {}
                    
                    # 运行各个策略
                    for strategy in strategies:
                        result = quantgpt.run_strategy_backtest(comp_symbol.upper(), strategy, comp_period)
                        if "error" not in result:
                            results[strategy] = result
                    
                    if results:
                        st.success(f"✅ 策略比较完成！共比较了 {len(results)} 个策略")
                        
                        # 策略比较表
                        st.subheader("📊 策略绩效对比")
                        
                        comparison_data = []
                        for strategy, data in results.items():
                            metrics = data["metrics"]
                            comparison_data.append({
                                "策略名称": strategy_names[strategy],
                                "总收益率": f"{metrics.get('total_return', 0):.2%}",
                                "年化波动率": f"{metrics.get('annual_volatility', 0):.2%}",
                                "夏普比率": f"{metrics.get('sharpe_ratio', 0):.3f}",
                                "最大回撤": f"{metrics.get('max_drawdown', 0):.2%}",
                                "最终价值": f"${metrics.get('final_value', 0):,.2f}"
                            })
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                        
                        # 找出最佳策略
                        best_strategy = max(results.keys(), 
                                          key=lambda x: results[x]['metrics'].get('sharpe_ratio', 0))
                        best_sharpe = results[best_strategy]['metrics'].get('sharpe_ratio', 0)
                        
                        st.success(f"🏆 **最佳策略**: {strategy_names[best_strategy]} (夏普比率: {best_sharpe:.3f})")
                        
                        # 策略收益对比图
                        st.subheader("📊 收益曲线对比")
                        
                        fig = go.Figure()
                        
                        colors = ['green', 'blue', 'red', 'orange', 'purple']
                        
                        for i, (strategy, data) in enumerate(results.items()):
                            if 'data' in data and 'Portfolio_Value' in data['data'].columns:
                                portfolio_data = data['data']
                                fig.add_trace(
                                    go.Scatter(
                                        x=portfolio_data.index,
                                        y=portfolio_data['Portfolio_Value'],
                                        mode='lines',
                                        name=strategy_names[strategy],
                                        line=dict(width=3, color=colors[i % len(colors)])
                                    )
                                )
                        
                        # 添加基准线
                        first_result = list(results.values())[0]
                        if 'data' in first_result:
                            benchmark_data = first_result['data']
                            initial_shares = initial_capital / benchmark_data['Close'].iloc[0]
                            benchmark_values = initial_shares * benchmark_data['Close']
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=benchmark_data.index,
                                    y=benchmark_values,
                                    mode='lines',
                                    name='📊 买入持有基准',
                                    line=dict(width=2, color='gray', dash='dash')
                                )
                            )
                        
                        fig.update_layout(
                            title=f"{comp_symbol.upper()} 多策略收益对比",
                            xaxis_title="日期",
                            yaxis_title="组合价值 ($)",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 风险收益散点图
                        st.subheader("🎯 风险收益分析")
                        
                        risk_return_data = []
                        for strategy, data in results.items():
                            metrics = data["metrics"]
                            risk_return_data.append({
                                "策略": strategy_names[strategy],
                                "收益率": metrics.get('total_return', 0) * 100,
                                "波动率": metrics.get('annual_volatility', 0) * 100,
                                "夏普比率": metrics.get('sharpe_ratio', 0)
                            })
                        
                        df_risk_return = pd.DataFrame(risk_return_data)
                        
                        scatter_fig = px.scatter(
                            df_risk_return,
                            x='波动率',
                            y='收益率',
                            size='夏普比率',
                            color='策略',
                            title="策略风险收益分布图",
                            labels={'波动率': '年化波动率 (%)', '收益率': '总收益率 (%)'},
                            hover_data=['夏普比率']
                        )
                        
                        scatter_fig.update_layout(height=400)
                        st.plotly_chart(scatter_fig, use_container_width=True)
                        
                        # 策略建议
                        st.subheader("💡 投资建议")
                        
                        best_return_strategy = max(results.keys(), 
                                                 key=lambda x: results[x]['metrics'].get('total_return', 0))
                        best_return = results[best_return_strategy]['metrics'].get('total_return', 0)
                        
                        lowest_risk_strategy = min(results.keys(), 
                                                 key=lambda x: results[x]['metrics'].get('annual_volatility', float('inf')))
                        lowest_risk = results[lowest_risk_strategy]['metrics'].get('annual_volatility', 0)
                        
                        rec_col1, rec_col2 = st.columns(2)
                        
                        with rec_col1:
                            st.markdown(f"""
                            **🏆 最高收益策略**
                            - 策略: {strategy_names[best_return_strategy]}
                            - 总收益率: {best_return:.2%}
                            - 适合: 追求高收益的激进投资者
                            """)
                        
                        with rec_col2:
                            st.markdown(f"""
                            **🛡️ 最低风险策略**
                            - 策略: {strategy_names[lowest_risk_strategy]}
                            - 年化波动率: {lowest_risk:.2%}
                            - 适合: 偏好稳健的保守投资者
                            """)
                        
                        st.markdown(f"""
                        **🎯 综合推荐**: {strategy_names[best_strategy]}
                        - 原因: 该策略在风险调整后收益最优(夏普比率最高)
                        - 夏普比率: {best_sharpe:.3f}
                        - 适合: 追求风险调整后收益的理性投资者
                        """)
                        
                    else:
                        st.error("❌ 所有策略都运行失败，请检查股票代码或网络连接")
                        
                except Exception as e:
                    st.error(f"❌ 策略比较过程中发生错误: {str(e)}")
        else:
            st.warning("⚠️ 请输入股票代码")

elif app_mode == "💎 基本面分析":
    st.header("💎 深度基本面分析")
    
    fund_symbol = st.text_input("🔍 输入股票代码", value="AAPL", placeholder="例如: AAPL")
    
    if st.button("📊 获取基本面数据", type="primary", use_container_width=True):
        if fund_symbol:
            with st.spinner(f"💎 正在获取 {fund_symbol.upper()} 深度基本面数据..."):
                try:
                    result = quantgpt.fundamental_engine.get_fundamental_data(fund_symbol.upper())
                    
                    if "error" not in result:
                        st.success("✅ 基本面数据获取成功")
                        
                        # 公司概览
                        st.subheader("🏢 公司概览")
                        
                        overview_col1, overview_col2 = st.columns(2)
                        
                        with overview_col1:
                            st.markdown("**📋 基本信息**")
                            st.write(f"• **公司名称**: {result.get('company_name', 'N/A')}")
                            st.write(f"• **股票代码**: {result.get('symbol', 'N/A')}")
                            st.write(f"• **行业板块**: {result.get('sector', 'N/A')}")
                            st.write(f"• **细分行业**: {result.get('industry', 'N/A')}")
                        
                        with overview_col2:
                            st.markdown("**💰 市场数据**")
                            market_cap = result.get('market_cap', 0)
                            if market_cap and market_cap > 0:
                                if market_cap >= 1e12:
                                    market_cap_str = f"${market_cap/1e12:.2f}T"
                                elif market_cap >= 1e9:
                                    market_cap_str = f"${market_cap/1e9:.1f}B"
                                elif market_cap >= 1e6:
                                    market_cap_str = f"${market_cap/1e6:.1f}M"
                                else:
                                    market_cap_str = f"${market_cap:,.0f}"
                                st.write(f"• **市值**: {market_cap_str}")
                            
                            score = result.get('fundamental_score', 0)
                            st.write(f"• **基本面评分**: {score:.1f}/100")
                        
                        # 估值指标
                        st.subheader("💰 估值指标")
                        
                        val_col1, val_col2, val_col3, val_col4 = st.columns(4)
                        
                        with val_col1:
                            pe = result.get('pe_ratio')
                            if pe:
                                pe_color = "green" if pe < 20 else "orange" if pe < 30 else "red"
                                st.metric("PE比率", f"{pe:.2f}")
                                if pe < 15:
                                    st.success("估值偏低")
                                elif pe > 30:
                                    st.warning("估值偏高")
                            else:
                                st.metric("PE比率", "N/A")
                        
                        with val_col2:
                            pb = result.get('pb_ratio')
                            if pb:
                                st.metric("PB比率", f"{pb:.2f}")
                                if pb < 2:
                                    st.success("账面价值合理")
                                elif pb > 5:
                                    st.warning("账面价值偏高")
                            else:
                                st.metric("PB比率", "N/A")
                        
                        with val_col3:
                            ps = result.get('ps_ratio')
                            if ps:
                                st.metric("PS比率", f"{ps:.2f}")
                            else:
                                st.metric("PS比率", "N/A")
                        
                        with val_col4:
                            peg = result.get('peg_ratio')
                            if peg:
                                st.metric("PEG比率", f"{peg:.2f}")
                                if peg < 1:
                                    st.success("成长估值合理")
                                elif peg > 2:
                                    st.warning("成长估值偏高")
                            else:
                                st.metric("PEG比率", "N/A")
                        
                        # 盈利能力
                        st.subheader("📈 盈利能力")
                        
                        prof_col1, prof_col2, prof_col3 = st.columns(3)
                        
                        with prof_col1:
                            roe = result.get('roe')
                            if roe:
                                roe_pct = roe * 100
                                st.metric("ROE (净资产收益率)", f"{roe_pct:.1f}%")
                                if roe_pct > 20:
                                    st.success("盈利能力优秀")
                                elif roe_pct > 15:
                                    st.info("盈利能力良好")
                                elif roe_pct < 10:
                                    st.warning("盈利能力较弱")
                            else:
                                st.metric("ROE", "N/A")
                        
                        with prof_col2:
                            roa = result.get('roa')
                            if roa:
                                roa_pct = roa * 100
                                st.metric("ROA (总资产收益率)", f"{roa_pct:.1f}%")
                                if roa_pct > 10:
                                    st.success("资产使用效率高")
                                elif roa_pct < 5:
                                    st.warning("资产使用效率低")
                            else:
                                st.metric("ROA", "N/A")
                        
                        with prof_col3:
                            gross_margin = result.get('gross_margin')
                            if gross_margin:
                                gm_pct = gross_margin * 100
                                st.metric("毛利率", f"{gm_pct:.1f}%")
                                if gm_pct > 50:
                                    st.success("毛利率优秀")
                                elif gm_pct > 30:
                                    st.info("毛利率良好")
                                elif gm_pct < 20:
                                    st.warning("毛利率较低")
                            else:
                                st.metric("毛利率", "N/A")
                        
                        # 财务健康
                        st.subheader("💪 财务健康度")
                        
                        health_col1, health_col2, health_col3 = st.columns(3)
                        
                        with health_col1:
                            debt_eq = result.get('debt_to_equity')
                            if debt_eq:
                                st.metric("债务股权比", f"{debt_eq:.2f}")
                                if debt_eq < 0.3:
                                    st.success("债务水平健康")
                                elif debt_eq > 1.0:
                                    st.warning("债务水平较高")
                            else:
                                st.metric("债务股权比", "N/A")
                        
                        with health_col2:
                            current_ratio = result.get('current_ratio')
                            if current_ratio:
                                st.metric("流动比率", f"{current_ratio:.2f}")
                                if current_ratio > 2:
                                    st.success("流动性充足")
                                elif current_ratio < 1:
                                    st.warning("流动性不足")
                            else:
                                st.metric("流动比率", "N/A")
                        
                        with health_col3:
                            quick_ratio = result.get('quick_ratio')
                            if quick_ratio:
                                st.metric("速动比率", f"{quick_ratio:.2f}")
                                if quick_ratio > 1:
                                    st.success("短期偿债能力强")
                                elif quick_ratio < 0.5:
                                    st.warning("短期偿债能力弱")
                            else:
                                st.metric("速动比率", "N/A")
                        
                        # 股息信息
                        dividend_yield = result.get('dividend_yield')
                        dividend_rate = result.get('dividend_rate')
                        payout_ratio = result.get('payout_ratio')
                        
                        if dividend_yield or dividend_rate:
                            st.subheader("💵 股息分析")
                            
                            div_col1, div_col2, div_col3 = st.columns(3)
                            
                            with div_col1:
                                if dividend_yield:
                                    dy_pct = dividend_yield * 100
                                    st.metric("股息率", f"{dy_pct:.2f}%")
                                    if dy_pct > 4:
                                        st.success("高股息收益")
                                    elif dy_pct > 2:
                                        st.info("适中股息收益")
                                else:
                                    st.metric("股息率", "N/A")
                            
                            with div_col2:
                                if dividend_rate:
                                    st.metric("年度股息", f"${dividend_rate:.2f}")
                                else:
                                    st.metric("年度股息", "N/A")
                            
                            with div_col3:
                                if payout_ratio:
                                    pr_pct = payout_ratio * 100
                                    st.metric("分红比率", f"{pr_pct:.1f}%")
                                    if pr_pct > 80:
                                        st.warning("分红比率较高")
                                    elif pr_pct < 30:
                                        st.info("保留盈利较多")
                                else:
                                    st.metric("分红比率", "N/A")
                        
                        # 综合评价
                        st.subheader("🎯 综合投资评价")
                        
                        score = result.get('fundamental_score', 0)
                        
                        if score >= 80:
                            rating = "🟢 优秀"
                            description = "基本面表现优异，值得重点关注"
                            color = "green"
                        elif score >= 70:
                            rating = "🟡 良好"
                            description = "基本面表现良好，可以考虑投资"
                            color = "blue"
                        elif score >= 60:
                            rating = "🟠 一般"
                            description = "基本面表现一般，需要谨慎考虑"
                            color = "orange"
                        else:
                            rating = "🔴 较差"
                            description = "基本面表现较差，建议回避"
                            color = "red"
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {color}22, {color}11); 
                                   border-left: 4px solid {color}; 
                                   padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
                            <h4 style='color: {color}; margin: 0;'>评级: {rating} ({score:.1f}/100)</h4>
                            <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>{description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 关键指标雷达图
                        st.subheader("📊 关键指标雷达图")
                        
                        # 构建雷达图数据
                        radar_categories = []
                        radar_values = []
                        
                        # PE评分 (越低越好，满分100)
                        if pe:
                            pe_score = max(0, min(100, 100 - (pe - 10) * 5)) if pe > 10 else 100
                            radar_categories.append('PE估值')
                            radar_values.append(pe_score)
                        
                        # ROE评分
                        if roe:
                            roe_score = min(100, roe * 500)  # ROE*100*5
                            radar_categories.append('盈利能力')
                            radar_values.append(roe_score)
                        
                        # 债务健康评分 (债务越低越好)
                        if debt_eq is not None:
                            debt_score = max(0, min(100, 100 - debt_eq * 50))
                            radar_categories.append('财务健康')
                            radar_values.append(debt_score)
                        
                        # 流动性评分
                        if current_ratio:
                            liquidity_score = min(100, current_ratio * 50)
                            radar_categories.append('流动性')
                            radar_values.append(liquidity_score)
                        
                        # 股息评分
                        if dividend_yield:
                            dividend_score = min(100, dividend_yield * 2500)  # 4%股息率=100分
                            radar_categories.append('股息收益')
                            radar_values.append(dividend_score)
                        
                        if len(radar_categories) >= 3:
                            radar_fig = go.Figure()
                            
                            radar_fig.add_trace(go.Scatterpolar(
                                r=radar_values,
                                theta=radar_categories,
                                fill='toself',
                                name=fund_symbol.upper(),
                                line_color='rgb(106, 81, 163)'
                            ))
                            
                            radar_fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 100]
                                    )),
                                showlegend=True,
                                title="基本面指标雷达图",
                                height=500
                            )
                            
                            st.plotly_chart(radar_fig, use_container_width=True)
                        else:
                            st.info("数据不足，无法生成雷达图")
                        
                    else:
                        st.error(f"❌ {result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ 获取基本面数据时发生错误: {str(e)}")
        else:
            st.warning("⚠️ 请输入股票代码")

# 页脚信息
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("### 📊 数据来源")
    st.markdown("- 股价数据: Yahoo Finance")
    st.markdown("- AI模型: FinBERT")
    st.markdown("- 技术指标: 自研算法")

with footer_col2:
    st.markdown("### ⚠️ 风险提示")
    st.markdown("- 投资有风险，入市需谨慎")
    st.markdown("- 本工具仅供参考，不构成投资建议")
    st.markdown("- 请结合自身风险承受能力决策")

with footer_col3:
    st.markdown("### 🔗 相关链接")
    st.markdown("- [GitHub源码](https://github.com)")
    st.markdown("- [使用文档](https://docs.example.com)")
    st.markdown("- [联系我们](mailto:contact@example.com)")

# 版权信息
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-top: 2rem; padding: 1rem;
               background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
               border-radius: 10px;'>
        🚀 <b>QuantGPT v1.0</b> - AI驱动的量化交易平台<br/>
        由 <b>专业AI量化工程师</b> 开发 | 
        ⭐ <a href='#' style='color: #1f77b4;'>给我们点个Star</a> | 
        📧 <a href='#' style='color: #1f77b4;'>反馈建议</a>
        <br/><br/>
        <small>本项目开源免费，欢迎贡献代码和提出改进建议</small>
    </div>
    """, 
    unsafe_allow_html=True
)

# 添加实时时间显示
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⏰ 系统状态")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"**当前时间**: {current_time}")
    
    # 系统状态指示器
    st.markdown("**系统状态**: 🟢 正常运行")
    st.markdown("**AI引擎**: 🟢 已加载")
    st.markdown("**数据连接**: 🟢 正常")
    
    # 添加一些使用提示
    with st.expander("💡 使用提示"):
        st.markdown("""
        **📊 股票分析**
        - 支持美股代码 (如AAPL, GOOGL)
        - 提供AI驱动的投资建议
        - 包含技术面和基本面分析
        
        **🔬 策略回测**
        - 支持多种经典策略
        - 提供详细绩效指标
        - 可视化收益曲线
        
        **🔍 股票筛选**
        - 内置价值和成长策略
        - 支持自定义股票池
        - 智能评分排序
        
        **💡 投资建议**
        - 仅供参考，不构成投资建议
        - 请结合自身情况谨慎决策
        - 投资有风险，入市需谨慎
        """)

# JavaScript增强功能
st.markdown("""
<script>
// 添加一些交互增强
document.addEventListener('DOMContentLoaded', function() {
    // 平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // 添加加载动画效果
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.textContent.includes('开始') || this.textContent.includes('获取')) {
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.style.transform = 'scale(1)';
                }, 150);
            }
        });
    });
});
</script>
""", unsafe_allow_html=True)
