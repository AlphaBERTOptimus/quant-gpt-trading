import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import re
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 页面配置 - 类似Claude.ai的简洁设计
st.set_page_config(
    page_title="QuantGPT - AI量化交易助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Claude.ai风格的CSS
st.markdown("""
<style>
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 主容器样式 */
    .main {
        padding-top: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* 聊天容器 */
    .chat-container {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        overflow: hidden;
    }
    
    /* 用户消息样式 */
    .user-message {
        background: #f7f7f8;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
    }
    
    /* AI消息样式 */
    .ai-message {
        background: white;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        border-left: 4px solid #10b981;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e5e7eb;
        padding: 12px 20px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    
    /* 头部样式 */
    .header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* 示例命令卡片 */
    .example-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .example-card:hover {
        background: #e2e8f0;
        transform: translateY(-1px);
    }
    
    /* 状态指示器 */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: #ecfdf5;
        border: 1px solid #d1fae5;
        border-radius: 20px;
        color: #065f46;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    
    /* 加载动画 */
    .thinking {
        padding: 1rem;
        text-align: center;
        color: #6b7280;
        font-style: italic;
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        
        .user-message, .ai-message {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 核心QuantGPT类定义
class QuantGPTCore:
    def __init__(self):
        self.cache = {}
        
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """获取股票数据"""
        cache_key = f"{symbol}_{period}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if not data.empty:
                self.cache[cache_key] = data
                return data
        except:
            pass
        return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """获取基本面数据"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "market_cap": info.get("marketCap"),
                "dividend_yield": info.get("dividendYield"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margins": info.get("profitMargins")
            }
        except:
            return {"symbol": symbol, "error": "无法获取数据"}
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """计算技术指标"""
        if data.empty:
            return {}
        
        close = data['Close']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 移动平均
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        
        # 布林带
        sma_20_bb = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_upper = sma_20_bb + (std_20 * 2)
        bb_lower = sma_20_bb - (std_20 * 2)
        
        return {
            "rsi": rsi.iloc[-1] if not rsi.empty else None,
            "sma_20": sma_20.iloc[-1] if not sma_20.empty else None,
            "sma_50": sma_50.iloc[-1] if not sma_50.empty else None,
            "macd": macd.iloc[-1] if not macd.empty else None,
            "macd_signal": signal.iloc[-1] if not signal.empty else None,
            "bb_upper": bb_upper.iloc[-1] if not bb_upper.empty else None,
            "bb_lower": bb_lower.iloc[-1] if not bb_lower.empty else None,
            "current_price": close.iloc[-1] if not close.empty else None,
            "volume": data['Volume'].iloc[-1] if 'Volume' in data.columns else None
        }

class NLPCommandParser:
    """自然语言命令解析器"""
    
    def __init__(self):
        self.patterns = {
            'analyze': [
                r'分析\s*([A-Z]{1,5})',
                r'帮我分析\s*([A-Z]{1,5})',
                r'analyze\s*([A-Z]{1,5})',
                r'给我看看\s*([A-Z]{1,5})',
                r'([A-Z]{1,5})\s*怎么样',
                r'([A-Z]{1,5})\s*的情况'
            ],
            'compare': [
                r'比较\s*([A-Z]{1,5})\s*和\s*([A-Z]{1,5})',
                r'([A-Z]{1,5})\s*vs\s*([A-Z]{1,5})',
                r'帮我比较\s*([A-Z]{1,5})\s*和\s*([A-Z]{1,5})',
                r'对比\s*([A-Z]{1,5})\s*和\s*([A-Z]{1,5})'
            ],
            'backtest': [
                r'回测\s*([A-Z]{1,5})\s*(.*?)策略',
                r'帮我回测\s*([A-Z]{1,5})',
                r'backtest\s*([A-Z]{1,5})',
                r'测试\s*([A-Z]{1,5})\s*的策略'
            ],
            'screen': [
                r'筛选.*?(PE|市盈率).*?([<>=]).*?(\d+\.?\d*)',
                r'筛选.*?(PB|市净率).*?([<>=]).*?(\d+\.?\d*)',
                r'筛选.*?(ROE).*?([<>=]).*?(\d+\.?\d*)',
                r'筛选.*?(RSI).*?([<>=]).*?(\d+\.?\d*)',
                r'筛选.*?(价格|股价).*?([<>=]).*?(\d+\.?\d*)',
                r'筛选.*?(市值).*?([<>=]).*?(\d+\.?\d*)',
                r'找.*?(高分红|分红).*?股票',
                r'找.*?(成长).*?股票',
                r'找.*?(价值).*?股票'
            ]
        }
    
    def parse_command(self, text: str) -> Dict:
        """解析自然语言命令"""
        text = text.upper().strip()
        
        # 分析单只股票
        for pattern in self.patterns['analyze']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    'action': 'analyze',
                    'symbol': match.group(1),
                    'confidence': 0.9
                }
        
        # 比较两只股票
        for pattern in self.patterns['compare']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    'action': 'compare',
                    'symbols': [match.group(1), match.group(2)],
                    'confidence': 0.9
                }
        
        # 回测策略
        for pattern in self.patterns['backtest']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                strategy = 'sma_crossover'  # 默认策略
                if 'RSI' in text:
                    strategy = 'rsi'
                elif 'MACD' in text:
                    strategy = 'macd'
                
                return {
                    'action': 'backtest',
                    'symbol': match.group(1),
                    'strategy': strategy,
                    'confidence': 0.8
                }
        
        # 股票筛选
        for pattern in self.patterns['screen']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if '高分红' in text or '分红' in text:
                    return {
                        'action': 'screen',
                        'type': 'dividend',
                        'confidence': 0.8
                    }
                elif '成长' in text:
                    return {
                        'action': 'screen',
                        'type': 'growth',
                        'confidence': 0.8
                    }
                elif '价值' in text:
                    return {
                        'action': 'screen',
                        'type': 'value',
                        'confidence': 0.8
                    }
                else:
                    # 具体指标筛选
                    indicator = match.group(1)
                    operator = match.group(2)
                    value = float(match.group(3))
                    
                    return {
                        'action': 'screen',
                        'type': 'custom',
                        'indicator': indicator,
                        'operator': operator,
                        'value': value,
                        'confidence': 0.9
                    }
        
        return {'action': 'unknown', 'confidence': 0.0}

class QuantGPTChatbot:
    """QuantGPT聊天机器人"""
    
    def __init__(self):
        self.core = QuantGPTCore()
        self.parser = NLPCommandParser()
        self.default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX", "JPM", "JNJ"]
    
    def analyze_stock(self, symbol: str) -> Dict:
        """分析单只股票"""
        # 获取数据
        data = self.core.get_stock_data(symbol)
        fundamental = self.core.get_fundamental_data(symbol)
        
        if data.empty:
            return {"error": f"无法获取 {symbol} 的数据"}
        
        technical = self.core.calculate_technical_indicators(data)
        
        # 生成分析结果
        current_price = technical.get('current_price', 0)
        rsi = technical.get('rsi', 50)
        
        # AI建议逻辑
        signals = []
        if rsi < 30:
            signals.append("RSI显示超卖，可能是买入机会")
        elif rsi > 70:
            signals.append("RSI显示超买，注意风险")
        
        if technical.get('sma_20', 0) > technical.get('sma_50', 0):
            signals.append("短期趋势向上")
        else:
            signals.append("短期趋势向下")
        
        # PE估值分析
        pe = fundamental.get('pe_ratio')
        if pe:
            if pe < 15:
                signals.append("PE估值偏低，可能被低估")
            elif pe > 30:
                signals.append("PE估值偏高，注意泡沫风险")
        
        return {
            "symbol": symbol,
            "name": fundamental.get('name', symbol),
            "current_price": current_price,
            "technical": technical,
            "fundamental": fundamental,
            "signals": signals,
            "recommendation": self._generate_recommendation(technical, fundamental)
        }
    
    def compare_stocks(self, symbols: List[str]) -> Dict:
        """比较股票"""
        results = {}
        for symbol in symbols:
            analysis = self.analyze_stock(symbol)
            if "error" not in analysis:
                results[symbol] = analysis
        
        if len(results) < 2:
            return {"error": "无法获取足够的股票数据进行比较"}
        
        # 生成比较结论
        comparison = self._generate_comparison(results)
        
        return {
            "symbols": symbols,
            "analyses": results,
            "comparison": comparison
        }
    
    def backtest_strategy(self, symbol: str, strategy: str = "sma_crossover") -> Dict:
        """回测策略"""
        data = self.core.get_stock_data(symbol, "2y")
        
        if data.empty:
            return {"error": f"无法获取 {symbol} 的历史数据"}
        
        # 简化的回测逻辑
        if strategy == "sma_crossover":
            return self._backtest_sma_crossover(symbol, data)
        elif strategy == "rsi":
            return self._backtest_rsi(symbol, data)
        else:
            return {"error": f"不支持的策略: {strategy}"}
    
    def screen_stocks(self, criteria: Dict) -> Dict:
        """筛选股票"""
        results = []
        
        for symbol in self.default_symbols:
            try:
                fundamental = self.core.get_fundamental_data(symbol)
                data = self.core.get_stock_data(symbol)
                
                if "error" in fundamental or data.empty:
                    continue
                
                technical = self.core.calculate_technical_indicators(data)
                
                # 应用筛选条件
                if self._meets_criteria(fundamental, technical, criteria):
                    results.append({
                        "symbol": symbol,
                        "name": fundamental.get('name', symbol),
                        "price": technical.get('current_price'),
                        "pe_ratio": fundamental.get('pe_ratio'),
                        "market_cap": fundamental.get('market_cap'),
                        "sector": fundamental.get('sector'),
                        "rsi": technical.get('rsi')
                    })
            except:
                continue
        
        # 按市值排序
        results.sort(key=lambda x: x.get('market_cap', 0) or 0, reverse=True)
        
        return {
            "criteria": criteria,
            "results": results,
            "count": len(results)
        }
    
    def _generate_recommendation(self, technical: Dict, fundamental: Dict) -> str:
        """生成投资建议"""
        score = 0
        
        # 技术面评分
        rsi = technical.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 1
        
        if technical.get('sma_20', 0) > technical.get('sma_50', 0):
            score += 1
        
        # 基本面评分
        pe = fundamental.get('pe_ratio')
        if pe and 10 <= pe <= 25:
            score += 1
        
        if fundamental.get('roe') and fundamental.get('roe') > 0.15:
            score += 1
        
        # 生成建议
        if score >= 3:
            return "🟢 建议买入 - 技术面和基本面都较为理想"
        elif score >= 2:
            return "🟡 可以关注 - 部分指标表现良好"
        else:
            return "🔴 建议观望 - 多项指标需要改善"
    
    def _generate_comparison(self, results: Dict) -> str:
        """生成比较结论"""
        symbols = list(results.keys())
        if len(symbols) != 2:
            return "比较数据不足"
        
        stock1, stock2 = symbols[0], symbols[1]
        data1, data2 = results[stock1], results[stock2]
        
        comparisons = []
        
        # 价格比较
        price1 = data1.get('current_price', 0)
        price2 = data2.get('current_price', 0)
        if price1 and price2:
            if price1 > price2:
                comparisons.append(f"{stock1} 价格更高 (${price1:.2f} vs ${price2:.2f})")
            else:
                comparisons.append(f"{stock2} 价格更高 (${price2:.2f} vs ${price1:.2f})")
        
        # PE比较
        pe1 = data1['fundamental'].get('pe_ratio')
        pe2 = data2['fundamental'].get('pe_ratio')
        if pe1 and pe2:
            if pe1 < pe2:
                comparisons.append(f"{stock1} PE更低，估值可能更合理 ({pe1:.1f} vs {pe2:.1f})")
            else:
                comparisons.append(f"{stock2} PE更低，估值可能更合理 ({pe2:.1f} vs {pe1:.1f})")
        
        # RSI比较
        rsi1 = data1['technical'].get('rsi', 50)
        rsi2 = data2['technical'].get('rsi', 50)
        if abs(rsi1 - 50) < abs(rsi2 - 50):
            comparisons.append(f"{stock1} RSI更接近中性区域，技术面更稳定")
        else:
            comparisons.append(f"{stock2} RSI更接近中性区域，技术面更稳定")
        
        return " | ".join(comparisons)
    
    def _backtest_sma_crossover(self, symbol: str, data: pd.DataFrame) -> Dict:
        """SMA交叉策略回测"""
        close = data['Close']
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        
        # 生成信号
        signals = (sma_20 > sma_50).astype(int)
        positions = signals.diff()
        
        # 简化的收益计算
        returns = close.pct_change()
        strategy_returns = signals.shift(1) * returns
        
        total_return = (1 + strategy_returns).prod() - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        return {
            "symbol": symbol,
            "strategy": "SMA交叉策略",
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "trades": len(positions[positions != 0])
        }
    
    def _backtest_rsi(self, symbol: str, data: pd.DataFrame) -> Dict:
        """RSI策略回测"""
        close = data['Close']
        
        # 计算RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        signals[rsi < 30] = 1  # 超卖买入
        signals[rsi > 70] = 0  # 超买卖出
        
        returns = close.pct_change()
        strategy_returns = signals.shift(1) * returns
        
        total_return = (1 + strategy_returns).prod() - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        return {
            "symbol": symbol,
            "strategy": "RSI策略",
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "trades": len(signals[signals.diff() != 0])
        }
    
    def _meets_criteria(self, fundamental: Dict, technical: Dict, criteria: Dict) -> bool:
        """检查是否满足筛选条件"""
        if criteria.get('type') == 'dividend':
            dividend_yield = fundamental.get('dividend_yield', 0)
            return dividend_yield and dividend_yield > 0.03
        
        elif criteria.get('type') == 'growth':
            revenue_growth = fundamental.get('revenue_growth', 0)
            return revenue_growth and revenue_growth > 0.15
        
        elif criteria.get('type') == 'value':
            pe = fundamental.get('pe_ratio', 0)
            return pe and pe < 20
        
        elif criteria.get('type') == 'custom':
            indicator = criteria.get('indicator')
            operator = criteria.get('operator')
            value = criteria.get('value')
            
            if indicator == 'PE':
                actual_value = fundamental.get('pe_ratio')
            elif indicator == 'PB':
                actual_value = fundamental.get('pb_ratio')
            elif indicator == 'RSI':
                actual_value = technical.get('rsi')
            elif indicator == '价格':
                actual_value = technical.get('current_price')
            else:
                return False
            
            if actual_value is None:
                return False
            
            if operator == '>':
                return actual_value > value
            elif operator == '<':
                return actual_value < value
            elif operator == '=':
                return abs(actual_value - value) < 0.1
            
        return True
    
    def process_command(self, text: str) -> Dict:
        """处理用户命令"""
        parsed = self.parser.parse_command(text)
        
        if parsed['action'] == 'analyze':
            return self.analyze_stock(parsed['symbol'])
        
        elif parsed['action'] == 'compare':
            return self.compare_stocks(parsed['symbols'])
        
        elif parsed['action'] == 'backtest':
            return self.backtest_strategy(parsed['symbol'], parsed.get('strategy', 'sma_crossover'))
        
        elif parsed['action'] == 'screen':
            return self.screen_stocks(parsed)
        
        else:
            return {
                "error": "抱歉，我没有理解您的指令。请尝试以下格式：",
                "examples": [
                    "分析 AAPL",
                    "比较 AAPL 和 GOOGL", 
                    "回测 TSLA 的RSI策略",
                    "筛选 PE < 20 的股票",
                    "找高分红的股票"
                ]
            }

# 初始化聊天机器人
@st.cache_resource
def get_chatbot():
    return QuantGPTChatbot()

def generate_ai_response(result: Dict) -> Dict:
    """生成AI回复"""
    response = {"role": "assistant", "content": ""}
    
    if "error" in result:
        if "examples" in result:
            response["content"] = f"""
{result['error']}

**示例命令：**
{chr(10).join(f"• {ex}" for ex in result['examples'])}

**支持的功能：**
- 📊 **股票分析**: 分析 [股票代码]
- ⚖️ **股票比较**: 比较 [股票1] 和 [股票2]  
- 🔬 **策略回测**: 回测 [股票代码] 的 [策略名] 策略
- 🔍 **股票筛选**: 筛选 [指标] [操作符] [数值] 的股票
- 🎯 **预设筛选**: 找高分红/成长/价值股票
"""
        else:
            response["content"] = result['error']
        return response
    
    # 股票分析结果
    if "current_price" in result:
        symbol = result['symbol']
        name = result['name']
        price = result['current_price']
        
        response["content"] = f"""
## 📊 {symbol} ({name}) 分析报告

### 💰 基本信息
- **当前价格**: ${price:.2f}
- **行业**: {result['fundamental'].get('sector', 'N/A')}
"""
        
        # 技术指标
        tech = result['technical']
        if tech:
            response["content"] += f"""
### 📈 技术指标
- **RSI**: {tech.get('rsi', 'N/A'):.1f} {'(超卖)' if tech.get('rsi', 50) < 30 else '(超买)' if tech.get('rsi', 50) > 70 else '(正常)'}
- **SMA 20**: ${tech.get('sma_20', 0):.2f}
- **SMA 50**: ${tech.get('sma_50', 0):.2f}
- **趋势**: {'看涨' if tech.get('sma_20', 0) > tech.get('sma_50', 0) else '看跌'}
"""
        
        # 基本面指标
        fund = result['fundamental']
        if fund.get('pe_ratio'):
            response["content"] += f"""
### 💎 基本面指标
- **PE比率**: {fund['pe_ratio']:.2f}
- **PB比率**: {fund.get('pb_ratio', 'N/A')}
- **ROE**: {fund.get('roe', 0)*100:.1f}% (如果有数据)
- **股息率**: {fund.get('dividend_yield', 0)*100:.2f}% (如果有数据)
"""
        
        # AI信号
        if result.get('signals'):
            response["content"] += f"""
### 🚨 关键信号
{chr(10).join(f"• {signal}" for signal in result['signals'])}
"""
        
        # 投资建议
        response["content"] += f"""
### 🎯 AI投资建议
{result.get('recommendation', '建议观望')}
"""
        
        # 生成价格图表
        data = st.session_state.chatbot.core.get_stock_data(symbol)
        if not data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=f'{symbol} 股价',
                line=dict(color='#2563eb', width=2)
            ))
            
            # 添加移动平均线
            if len(data) >= 20:
                sma_20 = data['Close'].rolling(20).mean()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=sma_20,
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ))
            
            fig.update_layout(
                title=f'{symbol} 股价走势图',
                xaxis_title='日期',
                yaxis_title='价格 ($)',
                height=400
            )
            
            response["chart_data"] = fig
    
    # 股票比较结果
    elif "comparison" in result:
        symbols = result['symbols']
        analyses = result['analyses']
        
        response["content"] = f"""
## ⚖️ {' vs '.join(symbols)} 对比分析

### 📊 对比结论
{result['comparison']}

### 📋 详细对比
"""
        
        # 创建对比表格
        comparison_data = []
        for symbol, analysis in analyses.items():
            comparison_data.append({
                "股票": symbol,
                "名称": analysis.get('name', symbol),
                "当前价格": f"${analysis.get('current_price', 0):.2f}",
                "PE比率": analysis['fundamental'].get('pe_ratio', 'N/A'),
                "RSI": f"{analysis['technical'].get('rsi', 50):.1f}",
                "建议": analysis.get('recommendation', '观望')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        response["table_data"] = comparison_df
        
        # 生成对比图表
        fig = go.Figure()
        for symbol, analysis in analyses.items():
            data = st.session_state.chatbot.core.get_stock_data(symbol)
            if not data.empty:
                # 标准化价格 (以第一天为基准)
                normalized_price = data['Close'] / data['Close'].iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized_price,
                    mode='lines',
                    name=f'{symbol}',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=f'{" vs ".join(symbols)} 价格表现对比 (标准化)',
            xaxis_title='日期',
            yaxis_title='相对表现 (%)',
            height=400
        )
        
        response["chart_data"] = fig
    
    # 回测结果
    elif "strategy" in result:
        symbol = result['symbol']
        strategy = result['strategy']
        total_return = result['total_return']
        sharpe = result['sharpe_ratio']
        
        response["content"] = f"""
## 🔬 {symbol} - {strategy} 回测报告

### 📈 绩效指标
- **总收益率**: {total_return:.2%}
- **年化波动率**: {result.get('volatility', 0):.2%}
- **夏普比率**: {sharpe:.3f}
- **交易次数**: {result.get('trades', 0)}

### 🎯 策略评估
"""
        
        if sharpe > 1:
            response["content"] += "🟢 **优秀策略** - 夏普比率>1，风险调整后收益良好"
        elif sharpe > 0.5:
            response["content"] += "🟡 **可接受策略** - 夏普比率>0.5，有一定投资价值"
        elif sharpe > 0:
            response["content"] += "🟠 **一般策略** - 夏普比率>0，但收益有限"
        else:
            response["content"] += "🔴 **不推荐策略** - 夏普比率<0，风险大于收益"
        
        response["content"] += f"""

### 📋 策略说明
"""
        if "SMA" in strategy:
            response["content"] += "**移动平均交叉策略**: 当短期均线(20日)上穿长期均线(50日)时买入，下穿时卖出"
        elif "RSI" in strategy:
            response["content"] += "**RSI策略**: 当RSI<30时买入(超卖)，RSI>70时卖出(超买)"
    
    # 筛选结果
    elif "results" in result:
        criteria = result['criteria']
        results = result['results']
        count = result['count']
        
        response["content"] = f"""
## 🔍 股票筛选结果

### 📊 筛选条件
"""
        
        if criteria.get('type') == 'dividend':
            response["content"] += "- **筛选类型**: 高分红股票 (股息率 > 3%)"
        elif criteria.get('type') == 'growth':
            response["content"] += "- **筛选类型**: 成长股票 (营收增长 > 15%)"
        elif criteria.get('type') == 'value':
            response["content"] += "- **筛选类型**: 价值股票 (PE < 20)"
        elif criteria.get('type') == 'custom':
            indicator = criteria.get('indicator')
            operator = criteria.get('operator')
            value = criteria.get('value')
            response["content"] += f"- **筛选条件**: {indicator} {operator} {value}"
        
        response["content"] += f"""

### 📋 筛选结果 (共找到 {count} 只股票)
"""
        
        if results:
            # 创建结果表格
            results_data = []
            for stock in results[:10]:  # 显示前10只
                results_data.append({
                    "股票代码": stock['symbol'],
                    "公司名称": stock.get('name', stock['symbol']),
                    "当前价格": f"${stock.get('price', 0):.2f}" if stock.get('price') else 'N/A',
                    "PE比率": f"{stock.get('pe_ratio', 0):.2f}" if stock.get('pe_ratio') else 'N/A',
                    "RSI": f"{stock.get('rsi', 50):.1f}" if stock.get('rsi') else 'N/A',
                    "行业": stock.get('sector', 'N/A')
                })
            
            results_df = pd.DataFrame(results_data)
            response["table_data"] = results_df
            
            if count > 10:
                response["content"] += f"\n*显示前10只股票，共筛选出{count}只符合条件的股票*"
        else:
            response["content"] += "\n❌ 未找到符合条件的股票，建议调整筛选条件"
    
    return response

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = get_chatbot()

# 头部
st.markdown("""
<div class="header">
    <h1>🤖 QuantGPT</h1>
    <p style="color: #6b7280; font-size: 1.1rem;">AI量化交易助手 - 专业的股票分析与交易策略平台</p>
</div>
""", unsafe_allow_html=True)

# 状态指示器
st.markdown("""
<div class="status-indicator">
    <span>🟢</span>
    <span>系统运行正常 | AI模型已加载 | 数据连接正常</span>
</div>
""", unsafe_allow_html=True)

# 示例命令 (如果没有历史消息)
if not st.session_state.messages:
    st.markdown("### 💡 试试这些命令:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 分析 AAPL", key="ex1"):
            st.session_state.messages.append({"role": "user", "content": "分析 AAPL"})
            st.rerun()
        
        if st.button("🔍 筛选 PE < 15 的股票", key="ex2"):
            st.session_state.messages.append({"role": "user", "content": "筛选 PE < 15 的股票"})
            st.rerun()
    
    with col2:
        if st.button("⚖️ 比较 AAPL 和 GOOGL", key="ex3"):
            st.session_state.messages.append({"role": "user", "content": "比较 AAPL 和 GOOGL"})
            st.rerun()
        
        if st.button("🔬 回测 TSLA 的RSI策略", key="ex4"):
            st.session_state.messages.append({"role": "user", "content": "回测 TSLA 的RSI策略"})
            st.rerun()

# 显示聊天历史
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>🧑‍💼 您:</strong><br/>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-message">
            <strong>🤖 QuantGPT:</strong><br/>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示AI回复内容
        if "chart_data" in message:
            # 显示图表
            st.plotly_chart(message["chart_data"], use_container_width=True)
        
        if "table_data" in message:
            # 显示表格
            st.dataframe(message["table_data"], use_container_width=True)
        
        # 显示文本内容
        st.markdown(message["content"])

# 输入框
st.markdown("### 💬 与QuantGPT对话")

# 创建输入框
user_input = st.text_input(
    "输入指令",
    placeholder="请输入您的指令，例如：分析 AAPL，比较 AAPL 和 GOOGL，筛选 PE < 20 的股票...",
    key="user_input",
    label_visibility="collapsed"
)

# 发送按钮
col1, col2 = st.columns([6, 1])
with col2:
    send_button = st.button("发送", type="primary", use_container_width=True)

# 处理用户输入
if send_button and user_input.strip():
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 显示思考状态
    with st.spinner("🤖 QuantGPT正在思考..."):
        try:
            # 处理命令
            result = st.session_state.chatbot.process_command(user_input)
            
            # 生成AI回复
            ai_response = generate_ai_response(result)
            
            # 添加AI消息
            st.session_state.messages.append(ai_response)
            
        except Exception as e:
            error_response = {
                "role": "assistant",
                "content": f"抱歉，处理您的请求时出现了错误：{str(e)}\n\n请检查网络连接或稍后重试。"
            }
            st.session_state.messages.append(error_response)
    
    # 清空输入框并刷新
    st.rerun()

# 清除对话历史按钮
if st.session_state.messages:
    if st.button("🗑️ 清除对话历史", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

# 侧边栏帮助信息
with st.sidebar:
    st.markdown("### 📚 使用指南")
    
    st.markdown("""
    **🔍 股票分析**
    - `分析 AAPL`
    - `帮我看看 GOOGL`
    - `TSLA 怎么样`
    
    **⚖️ 股票对比**
    - `比较 AAPL 和 GOOGL`
    - `TSLA vs NVDA`
    - `对比 META 和 NFLX`
    
    **🔬 策略回测**
    - `回测 AAPL 的SMA策略`
    - `测试 TSLA 的RSI策略`
    - `帮我回测 MSFT`
    
    **🎯 股票筛选**
    - `筛选 PE < 20 的股票`
    - `找 RSI > 70 的股票`
    - `筛选高分红股票`
    - `找成长股票`
    - `找价值股票`
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ 风险提示")
    st.markdown("""
    - 本工具仅供参考，不构成投资建议
    - 股市有风险，投资需谨慎
    - 请结合自身风险承受能力做决策
    - 历史表现不代表未来收益
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 数据来源")
    st.markdown("""
    - **股价数据**: Yahoo Finance
    - **技术指标**: 实时计算
    - **AI分析**: 基于量化模型
    - **更新频率**: 实时
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>🤖 QuantGPT v2.0</strong> - AI驱动的量化交易助手</p>
    <p>由专业量化团队开发 | 24/7 为您的投资决策提供智能支持</p>
    <p><small>⚠️ 投资有风险，本工具仅供参考，不构成投资建议</small></p>
</div>
""", unsafe_allow_html=True)
