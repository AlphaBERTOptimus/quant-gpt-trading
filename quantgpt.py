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
    page_title="QuantGPT - AI Quantitative Trading Assistant",
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
    """Natural Language Command Parser"""
    
    def __init__(self):
        self.patterns = {
            'analyze': [
                r'analyze\s*([A-Z]{1,5})',
                r'analyse\s*([A-Z]{1,5})',
                r'check\s*([A-Z]{1,5})',
                r'look\s*at\s*([A-Z]{1,5})',
                r'tell\s*me\s*about\s*([A-Z]{1,5})',
                r'what\s*about\s*([A-Z]{1,5})',
                r'how\s*is\s*([A-Z]{1,5})',
                r'([A-Z]{1,5})\s*analysis',
                r'show\s*me\s*([A-Z]{1,5})',
                r'give\s*me\s*([A-Z]{1,5})'
            ],
            'compare': [
                r'compare\s*([A-Z]{1,5})\s*(and|vs|versus|with)\s*([A-Z]{1,5})',
                r'([A-Z]{1,5})\s*vs\s*([A-Z]{1,5})',
                r'([A-Z]{1,5})\s*versus\s*([A-Z]{1,5})',
                r'difference\s*between\s*([A-Z]{1,5})\s*and\s*([A-Z]{1,5})'
            ],
            'backtest': [
                r'backtest\s*([A-Z]{1,5})\s*(.*?)strategy',
                r'test\s*([A-Z]{1,5})\s*strategy',
                r'run\s*backtest\s*on\s*([A-Z]{1,5})',
                r'backtest\s*([A-Z]{1,5})',
                r'test\s*([A-Z]{1,5})\s*(SMA|RSI|MACD)',
                r'strategy\s*test\s*([A-Z]{1,5})'
            ],
            'screen': [
                r'screen.*?(PE|P/E).*?([<>=]).*?(\d+\.?\d*)',
                r'screen.*?(PB|P/B).*?([<>=]).*?(\d+\.?\d*)',
                r'screen.*?(ROE).*?([<>=]).*?(\d+\.?\d*)',
                r'screen.*?(RSI).*?([<>=]).*?(\d+\.?\d*)',
                r'screen.*?(price).*?([<>=]).*?(\d+\.?\d*)',
                r'screen.*?(market\s*cap).*?([<>=]).*?(\d+\.?\d*)',
                r'find.*?(dividend|high\s*dividend).*?stocks',
                r'find.*?(growth).*?stocks',
                r'find.*?(value).*?stocks',
                r'screen\s*for.*?(dividend|growth|value)',
                r'show\s*me.*?(dividend|growth|value).*?stocks'
            ]
        }
    
    def parse_command(self, text: str) -> Dict:
        """Parse natural language command"""
        text = text.upper().strip()
        
        # Analyze single stock
        for pattern in self.patterns['analyze']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    'action': 'analyze',
                    'symbol': match.group(1),
                    'confidence': 0.9
                }
        
        # Compare two stocks
        for pattern in self.patterns['compare']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 3:  # pattern with 'and/vs/versus'
                    return {
                        'action': 'compare',
                        'symbols': [groups[0], groups[2]],
                        'confidence': 0.9
                    }
                else:  # pattern without middle word
                    return {
                        'action': 'compare',
                        'symbols': [groups[0], groups[1]],
                        'confidence': 0.9
                    }
        
        # Backtest strategy
        for pattern in self.patterns['backtest']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                strategy = 'sma_crossover'  # default strategy
                if 'RSI' in text:
                    strategy = 'rsi'
                elif 'MACD' in text:
                    strategy = 'macd'
                elif 'SMA' in text:
                    strategy = 'sma_crossover'
                
                return {
                    'action': 'backtest',
                    'symbol': match.group(1),
                    'strategy': strategy,
                    'confidence': 0.8
                }
        
        # Stock screening
        for pattern in self.patterns['screen']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'dividend' in text.lower() or 'high dividend' in text.lower():
                    return {
                        'action': 'screen',
                        'type': 'dividend',
                        'confidence': 0.8
                    }
                elif 'growth' in text.lower():
                    return {
                        'action': 'screen',
                        'type': 'growth',
                        'confidence': 0.8
                    }
                elif 'value' in text.lower():
                    return {
                        'action': 'screen',
                        'type': 'value',
                        'confidence': 0.8
                    }
                else:
                    # Specific indicator screening
                    groups = match.groups()
                    if len(groups) >= 3:
                        indicator = groups[0]
                        operator = groups[1]
                        value = float(groups[2])
                        
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
    """QuantGPT Chatbot"""
    
    def __init__(self):
        self.core = QuantGPTCore()
        self.parser = NLPCommandParser()
        self.default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX", "JPM", "JNJ"]
    
    def analyze_stock(self, symbol: str) -> Dict:
        """Analyze single stock"""
        # Get data
        data = self.core.get_stock_data(symbol)
        fundamental = self.core.get_fundamental_data(symbol)
        
        if data.empty:
            return {"error": f"Unable to fetch data for {symbol}"}
        
        technical = self.core.calculate_technical_indicators(data)
        
        # Generate analysis results
        current_price = technical.get('current_price', 0)
        rsi = technical.get('rsi', 50)
        
        # AI recommendation logic
        signals = []
        if rsi < 30:
            signals.append("RSI shows oversold condition, potential buy opportunity")
        elif rsi > 70:
            signals.append("RSI shows overbought condition, caution advised")
        
        if technical.get('sma_20', 0) > technical.get('sma_50', 0):
            signals.append("Short-term trend is bullish")
        else:
            signals.append("Short-term trend is bearish")
        
        # PE valuation analysis
        pe = fundamental.get('pe_ratio')
        if pe:
            if pe < 15:
                signals.append("PE ratio suggests undervaluation")
            elif pe > 30:
                signals.append("PE ratio suggests high valuation, bubble risk")
        
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
        """Compare stocks"""
        results = {}
        for symbol in symbols:
            analysis = self.analyze_stock(symbol)
            if "error" not in analysis:
                results[symbol] = analysis
        
        if len(results) < 2:
            return {"error": "Unable to fetch sufficient stock data for comparison"}
        
        # Generate comparison conclusion
        comparison = self._generate_comparison(results)
        
        return {
            "symbols": symbols,
            "analyses": results,
            "comparison": comparison
        }
    
    def backtest_strategy(self, symbol: str, strategy: str = "sma_crossover") -> Dict:
        """Backtest strategy"""
        data = self.core.get_stock_data(symbol, "2y")
        
        if data.empty:
            return {"error": f"Unable to fetch historical data for {symbol}"}
        
        # Simplified backtest logic
        if strategy == "sma_crossover":
            return self._backtest_sma_crossover(symbol, data)
        elif strategy == "rsi":
            return self._backtest_rsi(symbol, data)
        else:
            return {"error": f"Unsupported strategy: {strategy}"}
    
    def screen_stocks(self, criteria: Dict) -> Dict:
        """Screen stocks"""
        results = []
        
        for symbol in self.default_symbols:
            try:
                fundamental = self.core.get_fundamental_data(symbol)
                data = self.core.get_stock_data(symbol)
                
                if "error" in fundamental or data.empty:
                    continue
                
                technical = self.core.calculate_technical_indicators(data)
                
                # Apply screening criteria
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
        
        # Sort by market cap
        results.sort(key=lambda x: x.get('market_cap', 0) or 0, reverse=True)
        
        return {
            "criteria": criteria,
            "results": results,
            "count": len(results)
        }
    
    def _generate_recommendation(self, technical: Dict, fundamental: Dict) -> str:
        """Generate investment recommendation"""
        score = 0
        
        # Technical analysis scoring
        rsi = technical.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 1
        
        if technical.get('sma_20', 0) > technical.get('sma_50', 0):
            score += 1
        
        # Fundamental analysis scoring
        pe = fundamental.get('pe_ratio')
        if pe and 10 <= pe <= 25:
            score += 1
        
        if fundamental.get('roe') and fundamental.get('roe') > 0.15:
            score += 1
        
        # Generate recommendation
        if score >= 3:
            return "🟢 BUY - Both technical and fundamental indicators look favorable"
        elif score >= 2:
            return "🟡 HOLD - Some indicators show positive signals"
        else:
            return "🔴 SELL/AVOID - Multiple indicators need improvement"
    
    def _generate_comparison(self, results: Dict) -> str:
        """Generate comparison conclusion"""
        symbols = list(results.keys())
        if len(symbols) != 2:
            return "Insufficient comparison data"
        
        stock1, stock2 = symbols[0], symbols[1]
        data1, data2 = results[stock1], results[stock2]
        
        comparisons = []
        
        # Price comparison
        price1 = data1.get('current_price', 0)
        price2 = data2.get('current_price', 0)
        if price1 and price2:
            if price1 > price2:
                comparisons.append(f"{stock1} trades higher (${price1:.2f} vs ${price2:.2f})")
            else:
                comparisons.append(f"{stock2} trades higher (${price2:.2f} vs ${price1:.2f})")
        
        # PE comparison
        pe1 = data1['fundamental'].get('pe_ratio')
        pe2 = data2['fundamental'].get('pe_ratio')
        if pe1 and pe2:
            if pe1 < pe2:
                comparisons.append(f"{stock1} has lower PE, potentially better value ({pe1:.1f} vs {pe2:.1f})")
            else:
                comparisons.append(f"{stock2} has lower PE, potentially better value ({pe2:.1f} vs {pe1:.1f})")
        
        # RSI comparison
        rsi1 = data1['technical'].get('rsi', 50)
        rsi2 = data2['technical'].get('rsi', 50)
        if abs(rsi1 - 50) < abs(rsi2 - 50):
            comparisons.append(f"{stock1} RSI closer to neutral, more stable technically")
        else:
            comparisons.append(f"{stock2} RSI closer to neutral, more stable technically")
        
        return " | ".join(comparisons)
    
    def _backtest_sma_crossover(self, symbol: str, data: pd.DataFrame) -> Dict:
        """SMA crossover strategy backtest"""
        close = data['Close']
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        
        # Generate signals
        signals = (sma_20 > sma_50).astype(int)
        positions = signals.diff()
        
        # Simplified return calculation
        returns = close.pct_change()
        strategy_returns = signals.shift(1) * returns
        
        total_return = (1 + strategy_returns).prod() - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        return {
            "symbol": symbol,
            "strategy": "SMA Crossover Strategy",
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "trades": len(positions[positions != 0])
        }
    
    def _backtest_rsi(self, symbol: str, data: pd.DataFrame) -> Dict:
        """RSI strategy backtest"""
        close = data['Close']
        
        # Calculate RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[rsi < 30] = 1  # Oversold buy
        signals[rsi > 70] = 0  # Overbought sell
        
        returns = close.pct_change()
        strategy_returns = signals.shift(1) * returns
        
        total_return = (1 + strategy_returns).prod() - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        return {
            "symbol": symbol,
            "strategy": "RSI Strategy",
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "trades": len(signals[signals.diff() != 0])
        }
    
    def _meets_criteria(self, fundamental: Dict, technical: Dict, criteria: Dict) -> bool:
        """Check if meets screening criteria"""
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
            
            if indicator in ['PE', 'P/E']:
                actual_value = fundamental.get('pe_ratio')
            elif indicator in ['PB', 'P/B']:
                actual_value = fundamental.get('pb_ratio')
            elif indicator == 'RSI':
                actual_value = technical.get('rsi')
            elif indicator == 'PRICE':
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
        """Process user command"""
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
                "error": "Sorry, I didn't understand your command. Please try these formats:",
                "examples": [
                    "analyze AAPL",
                    "compare AAPL and GOOGL", 
                    "backtest TSLA RSI strategy",
                    "screen PE < 20",
                    "find dividend stocks"
                ]
            }

# 初始化聊天机器人
@st.cache_resource
def get_chatbot():
    return QuantGPTChatbot()

def generate_ai_response(result: Dict) -> Dict:
    """Generate AI response"""
    response = {"role": "assistant", "content": ""}
    
    if "error" in result:
        if "examples" in result:
            response["content"] = f"""
{result['error']}

**Example Commands:**
{chr(10).join(f"• {ex}" for ex in result['examples'])}

**Supported Features:**
- 📊 **Stock Analysis**: analyze [SYMBOL]
- ⚖️ **Stock Comparison**: compare [STOCK1] vs [STOCK2]  
- 🔬 **Strategy Backtest**: backtest [SYMBOL] [STRATEGY] strategy
- 🔍 **Stock Screening**: screen [INDICATOR] [OPERATOR] [VALUE]
- 🎯 **Preset Screening**: find dividend/growth/value stocks
"""
        else:
            response["content"] = result['error']
        return response
    
    # Stock analysis results
    if "current_price" in result:
        symbol = result['symbol']
        name = result['name']
        price = result['current_price']
        
        response["content"] = f"""
## 📊 {symbol} ({name}) Analysis Report

### 💰 Basic Information
- **Current Price**: ${price:.2f}
- **Sector**: {result['fundamental'].get('sector', 'N/A')}
"""
        
        # Technical indicators
        tech = result['technical']
        if tech:
            rsi_val = tech.get('rsi', 50)
            rsi_status = "(Oversold)" if rsi_val < 30 else "(Overbought)" if rsi_val > 70 else "(Normal)"
            trend = "Bullish" if tech.get('sma_20', 0) > tech.get('sma_50', 0) else "Bearish"
            
            response["content"] += f"""
### 📈 Technical Indicators
- **RSI**: {rsi_val:.1f} {rsi_status}
- **SMA 20**: ${tech.get('sma_20', 0):.2f}
- **SMA 50**: ${tech.get('sma_50', 0):.2f}
- **Trend**: {trend}
"""
        
        # Fundamental indicators
        fund = result['fundamental']
        if fund.get('pe_ratio'):
            response["content"] += f"""
### 💎 Fundamental Indicators
- **PE Ratio**: {fund['pe_ratio']:.2f}
- **PB Ratio**: {fund.get('pb_ratio', 'N/A')}
- **ROE**: {fund.get('roe', 0)*100:.1f}% (if available)
- **Dividend Yield**: {fund.get('dividend_yield', 0)*100:.2f}% (if available)
"""
        
        # AI signals
        if result.get('signals'):
            response["content"] += f"""
### 🚨 Key Signals
{chr(10).join(f"• {signal}" for signal in result['signals'])}
"""
        
        # Investment recommendation
        response["content"] += f"""
### 🎯 AI Investment Recommendation
{result.get('recommendation', 'Hold/Watch')}
"""
        
        # Generate price chart
        data = st.session_state.chatbot.core.get_stock_data(symbol)
        if not data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=f'{symbol} Price',
                line=dict(color='#2563eb', width=2)
            ))
            
            # Add moving averages
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
                title=f'{symbol} Price Chart',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=400
            )
            
            response["chart_data"] = fig
    
    # Stock comparison results
    elif "comparison" in result:
        symbols = result['symbols']
        analyses = result['analyses']
        
        response["content"] = f"""
## ⚖️ {' vs '.join(symbols)} Comparison Analysis

### 📊 Comparison Summary
{result['comparison']}

### 📋 Detailed Comparison
"""
        
        # Create comparison table
        comparison_data = []
        for symbol, analysis in analyses.items():
            comparison_data.append({
                "Stock": symbol,
                "Name": analysis.get('name', symbol),
                "Current Price": f"${analysis.get('current_price', 0):.2f}",
                "PE Ratio": analysis['fundamental'].get('pe_ratio', 'N/A'),
                "RSI": f"{analysis['technical'].get('rsi', 50):.1f}",
                "Recommendation": analysis.get('recommendation', 'Hold')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        response["table_data"] = comparison_df
        
        # Generate comparison chart
        fig = go.Figure()
        for symbol, analysis in analyses.items():
            data = st.session_state.chatbot.core.get_stock_data(symbol)
            if not data.empty:
                # Normalize prices (base = first day)
                normalized_price = data['Close'] / data['Close'].iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized_price,
                    mode='lines',
                    name=f'{symbol}',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=f'{" vs ".join(symbols)} Performance Comparison (Normalized)',
            xaxis_title='Date',
            yaxis_title='Relative Performance (%)',
            height=400
        )
        
        response["chart_data"] = fig
    
    # Backtest results
    elif "strategy" in result:
        symbol = result['symbol']
        strategy = result['strategy']
        total_return = result['total_return']
        sharpe = result['sharpe_ratio']
        
        response["content"] = f"""
## 🔬 {symbol} - {strategy} Backtest Report

### 📈 Performance Metrics
- **Total Return**: {total_return:.2%}
- **Annualized Volatility**: {result.get('volatility', 0):.2%}
- **Sharpe Ratio**: {sharpe:.3f}
- **Number of Trades**: {result.get('trades', 0)}

### 🎯 Strategy Evaluation
"""
        
        if sharpe > 1:
            response["content"] += "🟢 **Excellent Strategy** - Sharpe ratio >1, good risk-adjusted returns"
        elif sharpe > 0.5:
            response["content"] += "🟡 **Acceptable Strategy** - Sharpe ratio >0.5, has investment value"
        elif sharpe > 0:
            response["content"] += "🟠 **Average Strategy** - Sharpe ratio >0, but limited returns"
        else:
            response["content"] += "🔴 **Not Recommended** - Sharpe ratio <0, risk exceeds returns"
        
        response["content"] += f"""

### 📋 Strategy Description
"""
        if "SMA" in strategy:
            response["content"] += "**SMA Crossover Strategy**: Buy when short-term MA (20-day) crosses above long-term MA (50-day), sell when it crosses below"
        elif "RSI" in strategy:
            response["content"] += "**RSI Strategy**: Buy when RSI<30 (oversold), sell when RSI>70 (overbought)"
    
    # Screening results
    elif "results" in result:
        criteria = result['criteria']
        results = result['results']
        count = result['count']
        
        response["content"] = f"""
## 🔍 Stock Screening Results

### 📊 Screening Criteria
"""
        
        if criteria.get('type') == 'dividend':
            response["content"] += "- **Screen Type**: High Dividend Stocks (Dividend Yield > 3%)"
        elif criteria.get('type') == 'growth':
            response["content"] += "- **Screen Type**: Growth Stocks (Revenue Growth > 15%)"
        elif criteria.get('type') == 'value':
            response["content"] += "- **Screen Type**: Value Stocks (PE < 20)"
        elif criteria.get('type') == 'custom':
            indicator = criteria.get('indicator')
            operator = criteria.get('operator')
            value = criteria.get('value')
            response["content"] += f"- **Screen Condition**: {indicator} {operator} {value}"
        
        response["content"] += f"""

### 📋 Screening Results (Found {count} stocks)
"""
        
        if results:
            # Create results table
            results_data = []
            for stock in results[:10]:  # Show top 10
                results_data.append({
                    "Symbol": stock['symbol'],
                    "Company Name": stock.get('name', stock['symbol']),
                    "Current Price": f"${stock.get('price', 0):.2f}" if stock.get('price') else 'N/A',
                    "PE Ratio": f"{stock.get('pe_ratio', 0):.2f}" if stock.get('pe_ratio') else 'N/A',
                    "RSI": f"{stock.get('rsi', 50):.1f}" if stock.get('rsi') else 'N/A',
                    "Sector": stock.get('sector', 'N/A')
                })
            
            results_df = pd.DataFrame(results_data)
            response["table_data"] = results_df
            
            if count > 10:
                response["content"] += f"\n*Showing top 10 stocks, found {count} stocks matching criteria*"
        else:
            response["content"] += "\n❌ No stocks found matching criteria, consider adjusting filters"
    
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
    <p style="color: #6b7280; font-size: 1.1rem;">AI Quantitative Trading Assistant - Professional Stock Analysis & Trading Strategy Platform</p>
</div>
""", unsafe_allow_html=True)

# 状态指示器
st.markdown("""
<div class="status-indicator">
    <span>🟢</span>
    <span>System Online | AI Model Loaded | Data Connection Active</span>
</div>
""", unsafe_allow_html=True)

# 示例命令 (如果没有历史消息)
if not st.session_state.messages:
    st.markdown("### 💡 Try these commands:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Analyze AAPL", key="ex1"):
            st.session_state.messages.append({"role": "user", "content": "analyze AAPL"})
            st.rerun()
        
        if st.button("🔍 Screen PE < 15 stocks", key="ex2"):
            st.session_state.messages.append({"role": "user", "content": "screen PE < 15"})
            st.rerun()
    
    with col2:
        if st.button("⚖️ Compare AAPL vs GOOGL", key="ex3"):
            st.session_state.messages.append({"role": "user", "content": "compare AAPL vs GOOGL"})
            st.rerun()
        
        if st.button("🔬 Backtest TSLA RSI strategy", key="ex4"):
            st.session_state.messages.append({"role": "user", "content": "backtest TSLA RSI strategy"})
            st.rerun()

# 显示聊天历史
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 You:</strong><br/>
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
st.markdown("### 💬 Chat with QuantGPT")

# 创建输入框
user_input = st.text_input(
    "Enter Command",
    placeholder="Enter your command, e.g.: analyze AAPL, compare AAPL vs GOOGL, screen PE < 20...",
    key="user_input",
    label_visibility="collapsed"
)

# 发送按钮
col1, col2 = st.columns([6, 1])
with col2:
    send_button = st.button("Send", type="primary", use_container_width=True)

# 处理用户输入
if send_button and user_input.strip():
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 显示思考状态
    with st.spinner("🤖 QuantGPT is thinking..."):
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
                "content": f"Sorry, an error occurred while processing your request: {str(e)}\n\nPlease check your network connection or try again later."
            }
            st.session_state.messages.append(error_response)
    
    # 清空输入框并刷新
    st.rerun()

# 清除对话历史按钮
if st.session_state.messages:
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

# 侧边栏帮助信息
with st.sidebar:
    st.markdown("### 📚 User Guide")
    
    st.markdown("""
    **🔍 Stock Analysis**
    - `analyze AAPL`
    - `tell me about GOOGL`
    - `how is TSLA`
    
    **⚖️ Stock Comparison**
    - `compare AAPL and GOOGL`
    - `TSLA vs NVDA`
    - `compare META with NFLX`
    
    **🔬 Strategy Backtest**
    - `backtest AAPL SMA strategy`
    - `test TSLA RSI strategy`
    - `backtest MSFT`
    
    **🎯 Stock Screening**
    - `screen PE < 20`
    - `find RSI > 70 stocks`
    - `screen dividend stocks`
    - `find growth stocks`
    - `find value stocks`
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Risk Disclaimer")
    st.markdown("""
    - This tool is for reference only, not investment advice
    - Stock markets involve risks, invest cautiously
    - Consider your risk tolerance before making decisions
    - Past performance doesn't guarantee future returns
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Data Sources")
    st.markdown("""
    - **Stock Data**: Yahoo Finance
    - **Technical Indicators**: Real-time calculation
    - **AI Analysis**: Based on quantitative models
    - **Update Frequency**: Real-time
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>🤖 QuantGPT v2.0</strong> - AI-Powered Quantitative Trading Assistant</p>
    <p>Developed by Professional Quant Team | 24/7 Intelligent Support for Your Investment Decisions</p>
    <p><small>⚠️ Investment involves risks. This tool is for reference only and does not constitute investment advice.</small></p>
</div>
""", unsafe_allow_html=True)
