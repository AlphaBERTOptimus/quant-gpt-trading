import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import re
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 专业交易界面配置
st.set_page_config(
    page_title="QuantGPT Pro - US Stock Trading Terminal",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 专业交易终端CSS样式
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');
    
    /* 专业深色主题 */
    .stApp {
        background: linear-gradient(135deg, #0c1426 0%, #1a1f36 50%, #0c1426 100%);
        color: #e0e6ed;
    }
    
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* 主容器 */
    .main {
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* 专业标题 */
    .pro-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #3b82f6;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
    }
    
    .terminal-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 0;
        text-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        letter-spacing: 2px;
    }
    
    .terminal-subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* 状态栏 */
    .status-bar {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #10b981;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }
    
    /* 专业消息框 */
    .user-message {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .ai-message {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #10b981;
        border-left: 4px solid #10b981;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2);
    }
    
    /* 专业输入框 */
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.9);
        border: 2px solid #475569;
        border-radius: 8px;
        color: #e0e6ed;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
        background: rgba(15, 23, 42, 1);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #64748b;
    }
    
    /* 专业按钮 */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border: 2px solid #3b82f6;
        border-radius: 8px;
        color: white;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* 示例命令卡片 */
    .command-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .command-card:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    
    /* 专业侧边栏 */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid #334155;
    }
    
    /* 专业数据表格 */
    .stDataFrame {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 8px;
        border: 1px solid #334155;
    }
    
    /* 专业选择框 */
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid #475569;
        border-radius: 6px;
        color: #e0e6ed;
    }
    
    /* 专业指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0.2rem 0 0 0;
    }
    
    /* 专业标签 */
    .pro-badge {
        display: inline-block;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .terminal-title {
            font-size: 2rem;
        }
        .main {
            padding: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 扩展的技术指标计算器
class AdvancedTechnicalIndicators:
    """高级技术指标计算器"""
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> Dict:
        """计算所有技术指标"""
        if data.empty:
            return {}
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        indicators = {}
        
        # 基础移动平均
        for period in [5, 10, 20, 50, 100, 200]:
            indicators[f'SMA_{period}'] = close.rolling(period).mean().iloc[-1] if len(close) >= period else None
            indicators[f'EMA_{period}'] = close.ewm(span=period).mean().iloc[-1] if len(close) >= period else None
        
        # RSI系列
        for period in [14, 21]:
            rsi = AdvancedTechnicalIndicators._calculate_rsi(close, period)
            indicators[f'RSI_{period}'] = rsi.iloc[-1] if not rsi.empty else None
        
        # MACD
        macd_data = AdvancedTechnicalIndicators._calculate_macd(close)
        indicators.update(macd_data)
        
        # 布林带
        bb_data = AdvancedTechnicalIndicators._calculate_bollinger_bands(close)
        indicators.update(bb_data)
        
        # KDJ指标
        kdj_data = AdvancedTechnicalIndicators._calculate_kdj(high, low, close)
        indicators.update(kdj_data)
        
        # 威廉指标
        indicators['Williams_R'] = AdvancedTechnicalIndicators._calculate_williams_r(high, low, close)
        
        # CCI指标
        indicators['CCI'] = AdvancedTechnicalIndicators._calculate_cci(high, low, close)
        
        # ATR波动率
        indicators['ATR'] = AdvancedTechnicalIndicators._calculate_atr(high, low, close)
        
        # 成交量指标
        indicators['Volume_SMA'] = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else None
        indicators['Volume_Ratio'] = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else None
        
        # 价格基础数据
        indicators['Current_Price'] = close.iloc[-1]
        indicators['Price_Change'] = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100 if len(close) > 1 else 0
        indicators['High_52W'] = close.rolling(252).max().iloc[-1] if len(close) >= 252 else close.max()
        indicators['Low_52W'] = close.rolling(252).min().iloc[-1] if len(close) >= 252 else close.min()
        
        return indicators
    
    @staticmethod
    def _calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """计算MACD"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'MACD': macd.iloc[-1] if not macd.empty else None,
            'MACD_Signal': signal_line.iloc[-1] if not signal_line.empty else None,
            'MACD_Histogram': histogram.iloc[-1] if not histogram.empty else None
        }
    
    @staticmethod
    def _calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict:
        """计算布林带"""
        sma = data.rolling(window).mean()
        std = data.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return {
            'BB_Upper': upper.iloc[-1] if not upper.empty else None,
            'BB_Middle': sma.iloc[-1] if not sma.empty else None,
            'BB_Lower': lower.iloc[-1] if not lower.empty else None,
            'BB_Width': ((upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1] * 100) if not upper.empty and not lower.empty and sma.iloc[-1] != 0 else None
        }
    
    @staticmethod
    def _calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 9) -> Dict:
        """计算KDJ指标"""
        if len(close) < window:
            return {'K': None, 'D': None, 'J': None}
        
        lowest_low = low.rolling(window).min()
        highest_high = high.rolling(window).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(0)
        
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d
        
        return {
            'K': k.iloc[-1] if not k.empty else None,
            'D': d.iloc[-1] if not d.empty else None,
            'J': j.iloc[-1] if not j.empty else None
        }
    
    @staticmethod
    def _calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
        """计算威廉指标"""
        if len(close) < window:
            return None
        
        highest_high = high.rolling(window).max()
        lowest_low = low.rolling(window).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.iloc[-1] if not wr.empty else None
    
    @staticmethod
    def _calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> float:
        """计算CCI指标"""
        if len(close) < window:
            return None
        
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window).mean()
        mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci.iloc[-1] if not cci.empty else None
    
    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
        """计算ATR"""
        if len(close) < 2:
            return None
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window).mean()
        return atr.iloc[-1] if not atr.empty else None

# 扩展基本面分析引擎
class ComprehensiveFundamentalAnalysis:
    """全面基本面分析"""
    
    def __init__(self):
        self.cache = {}
    
    def get_comprehensive_fundamentals(self, symbol: str) -> Dict:
        """获取全面基本面数据"""
        if symbol in self.cache:
            return self.cache[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 基础信息
            fundamentals = {
                "symbol": symbol,
                "company_name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "country": info.get("country", "US"),
                "website": info.get("website", ""),
                "business_summary": info.get("longBusinessSummary", "")[:500] + "..." if info.get("longBusinessSummary") else "",
                
                # 估值指标
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "peg_ratio": info.get("pegRatio"),
                "ev_revenue": info.get("enterpriseToRevenue"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                
                # 盈利能力
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "roic": info.get("returnOnCapital"),
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "net_margin": info.get("profitMargins"),
                "ebitda_margin": info.get("ebitdaMargins"),
                
                # 财务健康
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "cash_ratio": info.get("cashRatio"),
                "interest_coverage": info.get("interestCoverage"),
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "free_cash_flow": info.get("freeCashflow"),
                
                # 成长性
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "revenue_per_share": info.get("revenuePerShare"),
                "earnings_per_share": info.get("trailingEps"),
                "book_value_per_share": info.get("bookValue"),
                
                # 股息
                "dividend_yield": info.get("dividendYield"),
                "dividend_rate": info.get("dividendRate"),
                "payout_ratio": info.get("payoutRatio"),
                "dividend_date": info.get("dividendDate"),
                "ex_dividend_date": info.get("exDividendDate"),
                
                # 市场数据
                "beta": info.get("beta"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
                "shares_short": info.get("sharesShort"),
                "short_ratio": info.get("shortRatio"),
                "insider_ownership": info.get("heldPercentInsiders"),
                "institutional_ownership": info.get("heldPercentInstitutions"),
                
                # 分析师预期
                "target_mean_price": info.get("targetMeanPrice"),
                "target_high_price": info.get("targetHighPrice"),
                "target_low_price": info.get("targetLowPrice"),
                "recommendation_mean": info.get("recommendationMean"),
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
                
                # 业务指标
                "price_to_sales_ttm": info.get("priceToSalesTrailing12Months"),
                "enterprise_value_revenue": info.get("enterpriseToRevenue"),
                "profit_margins": info.get("profitMargins"),
                "operating_cash_flow": info.get("operatingCashflow"),
                "levered_free_cash_flow": info.get("freeCashflow"),
                
                "last_updated": datetime.now().isoformat()
            }
            
            # 计算衍生指标
            fundamentals.update(self._calculate_derived_scores(fundamentals))
            
            self.cache[symbol] = fundamentals
            return fundamentals
            
        except Exception as e:
            return {"symbol": symbol, "error": f"获取基本面数据失败: {str(e)}"}
    
    def _calculate_derived_scores(self, data: Dict) -> Dict:
        """计算衍生评分"""
        scores = {}
        
        # 价值评分 (0-100)
        value_score = 50
        pe = data.get("pe_ratio")
        if pe:
            if pe < 10: value_score += 20
            elif pe < 15: value_score += 15
            elif pe < 20: value_score += 10
            elif pe > 30: value_score -= 15
        
        pb = data.get("pb_ratio")
        if pb:
            if pb < 1: value_score += 15
            elif pb < 2: value_score += 10
            elif pb > 5: value_score -= 10
        
        scores["value_score"] = max(0, min(100, value_score))
        
        # 质量评分
        quality_score = 50
        roe = data.get("roe")
        if roe:
            if roe > 0.25: quality_score += 20
            elif roe > 0.20: quality_score += 15
            elif roe > 0.15: quality_score += 10
            elif roe < 0.10: quality_score -= 10
        
        debt_eq = data.get("debt_to_equity")
        if debt_eq is not None:
            if debt_eq < 0.3: quality_score += 15
            elif debt_eq < 0.5: quality_score += 10
            elif debt_eq > 1.5: quality_score -= 15
        
        scores["quality_score"] = max(0, min(100, quality_score))
        
        # 成长评分
        growth_score = 50
        rev_growth = data.get("revenue_growth")
        if rev_growth:
            if rev_growth > 0.3: growth_score += 25
            elif rev_growth > 0.2: growth_score += 20
            elif rev_growth > 0.15: growth_score += 15
            elif rev_growth > 0.1: growth_score += 10
            elif rev_growth < 0: growth_score -= 20
        
        scores["growth_score"] = max(0, min(100, growth_score))
        
        # 综合评分
        scores["overall_score"] = (scores["value_score"] * 0.3 + 
                                 scores["quality_score"] * 0.4 + 
                                 scores["growth_score"] * 0.3)
        
        return scores

# 美股专用命令解析器
class USStockCommandParser:
    """美股专用命令解析器"""
    
    def __init__(self):
        self.patterns = {
            'analyze': [
                # 英文
                r'analyze\s*([A-Z][A-Z0-9]{0,9})',
                r'analyse\s*([A-Z][A-Z0-9]{0,9})',
                r'check\s*([A-Z][A-Z0-9]{0,9})',
                r'look\s*at\s*([A-Z][A-Z0-9]{0,9})',
                r'tell\s*me\s*about\s*([A-Z][A-Z0-9]{0,9})',
                r'show\s*me\s*([A-Z][A-Z0-9]{0,9})',
                r'([A-Z][A-Z0-9]{0,9})\s*analysis',
                # 中文
                r'分析\s*([A-Z][A-Z0-9]{0,9})',
                r'查看\s*([A-Z][A-Z0-9]{0,9})',
                r'看看\s*([A-Z][A-Z0-9]{0,9})',
                r'([A-Z][A-Z0-9]{0,9})\s*怎么样',
                r'([A-Z][A-Z0-9]{0,9})\s*分析',
                r'帮我分析\s*([A-Z][A-Z0-9]{0,9})',
            ],
            'compare': [
                # 英文
                r'compare\s*([A-Z][A-Z0-9]{0,9})\s*(and|vs|versus|with)\s*([A-Z][A-Z0-9]{0,9})',
                r'([A-Z][A-Z0-9]{0,9})\s*vs\s*([A-Z][A-Z0-9]{0,9})',
                r'([A-Z][A-Z0-9]{0,9})\s*versus\s*([A-Z][A-Z0-9]{0,9})',
                # 中文
                r'比较\s*([A-Z][A-Z0-9]{0,9})\s*(和|与)\s*([A-Z][A-Z0-9]{0,9})',
                r'([A-Z][A-Z0-9]{0,9})\s*对比\s*([A-Z][A-Z0-9]{0,9})',
                r'对比\s*([A-Z][A-Z0-9]{0,9})\s*(和|与)\s*([A-Z][A-Z0-9]{0,9})',
            ],
            'backtest': [
                # 英文
                r'backtest\s*([A-Z][A-Z0-9]{0,9})\s*(.*?)strategy',
                r'test\s*([A-Z][A-Z0-9]{0,9})\s*strategy',
                r'backtest\s*([A-Z][A-Z0-9]{0,9})',
                # 中文
                r'回测\s*([A-Z][A-Z0-9]{0,9})\s*(.*?)策略',
                r'测试\s*([A-Z][A-Z0-9]{0,9})\s*策略',
                r'回测\s*([A-Z][A-Z0-9]{0,9})',
            ],
            'screen': [
                # 英文
                r'screen.*?(PE|P/E|市盈率).*?([<>=]).*?(\d+\.?\d*)',
                r'screen.*?(PB|P/B|市净率).*?([<>=]).*?(\d+\.?\d*)',
                r'screen.*?(ROE).*?([<>=]).*?(\d+\.?\d*)',
                r'find.*?(dividend|growth|value).*?stocks',
                # 中文
                r'筛选.*?(PE|P/E|市盈率).*?([<>=]).*?(\d+\.?\d*)',
                r'筛选.*?(PB|P/B|市净率).*?([<>=]).*?(\d+\.?\d*)',
                r'筛选.*?(ROE|净资产收益率).*?([<>=]).*?(\d+\.?\d*)',
                r'找.*?(高分红|成长|价值).*?股票',
                r'寻找.*?(分红|成长股|价值股)',
            ]
        }
    
    def parse_command(self, text: str) -> Dict:
        """解析美股命令"""
        text = text.upper().strip()
        
        # 分析命令
        for pattern in self.patterns['analyze']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                symbol = match.group(1).upper()
                # 验证是否为有效美股代码格式
                if self._is_valid_us_stock_symbol(symbol):
                    return {
                        'action': 'analyze',
                        'symbol': symbol,
                        'confidence': 0.9
                    }
        
        # 比较命令
        for pattern in self.patterns['compare']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    symbol1, symbol2 = groups[0].upper(), groups[2].upper()
                    if self._is_valid_us_stock_symbol(symbol1) and self._is_valid_us_stock_symbol(symbol2):
                        return {
                            'action': 'compare',
                            'symbols': [symbol1, symbol2],
                            'confidence': 0.9
                        }
                elif len(groups) >= 2:
                    symbol1, symbol2 = groups[0].upper(), groups[1].upper()
                    if self._is_valid_us_stock_symbol(symbol1) and self._is_valid_us_stock_symbol(symbol2):
                        return {
                            'action': 'compare',
                            'symbols': [symbol1, symbol2],
                            'confidence': 0.9
                        }
        
        # 回测命令
        for pattern in self.patterns['backtest']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                symbol = match.group(1).upper()
                if self._is_valid_us_stock_symbol(symbol):
                    strategy = 'sma_crossover'
                    if any(x in text.upper() for x in ['RSI', 'rsi']):
                        strategy = 'rsi'
                    elif any(x in text.upper() for x in ['MACD', 'macd']):
                        strategy = 'macd'
                    
                    return {
                        'action': 'backtest',
                        'symbol': symbol,
                        'strategy': strategy,
                        'confidence': 0.8
                    }
        
        # 筛选命令
        for pattern in self.patterns['screen']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if any(x in text.lower() for x in ['dividend', '分红', '高分红']):
                    return {
                        'action': 'screen',
                        'type': 'dividend',
                        'confidence': 0.8
                    }
                elif any(x in text.lower() for x in ['growth', '成长', '成长股']):
                    return {
                        'action': 'screen',
                        'type': 'growth',
                        'confidence': 0.8
                    }
                elif any(x in text.lower() for x in ['value', '价值', '价值股']):
                    return {
                        'action': 'screen',
                        'type': 'value',
                        'confidence': 0.8
                    }
                else:
                    # 具体指标筛选
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
    
    def _is_valid_us_stock_symbol(self, symbol: str) -> bool:
        """验证是否为有效的美股代码格式"""
        if not symbol or len(symbol) < 1 or len(symbol) > 10:
            return False
        
        # 美股代码通常是1-10个字母，可能包含少量数字
        # 第一个字符必须是字母
        if not symbol[0].isalpha():
            return False
        
        # 其余字符可以是字母或数字
        for char in symbol[1:]:
            if not (char.isalpha() or char.isdigit()):
                return False
        
        return True

# 美股数据管理器
class USStockDataManager:
    """美股数据管理器"""
    
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """获取美股数据"""
        # 确保是大写
        symbol = symbol.upper().strip()
        
        cache_key = f"{symbol}_{period}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                self.cache[cache_key] = data
                return data
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
        
        return pd.DataFrame()

# 专业美股量化分析引擎
class USStockQuantEngine:
    """专业美股量化分析引擎"""
    
    def __init__(self):
        self.data_manager = USStockDataManager()
        self.technical_analyzer = AdvancedTechnicalIndicators()
        self.fundamental_analyzer = ComprehensiveFundamentalAnalysis()
        self.command_parser = USStockCommandParser()
    
    def comprehensive_analysis(self, symbol: str, period: str = "1y") -> Dict:
        """全面分析"""
        # 获取数据
        data = self.data_manager.get_stock_data(symbol, period)
        if data.empty:
            return {"error": f"无法获取 {symbol} 的数据，请检查股票代码是否正确"}
        
        # 技术分析
        technical_indicators = self.technical_analyzer.calculate_all_indicators(data)
        
        # 基本面分析
        fundamental_data = self.fundamental_analyzer.get_comprehensive_fundamentals(symbol)
        
        # 生成AI洞察
        ai_insights = self._generate_ai_insights(technical_indicators, fundamental_data)
        
        return {
            "symbol": symbol,
            "company_info": {
                "name": fundamental_data.get("company_name", symbol),
                "sector": fundamental_data.get("sector", "Unknown"),
                "industry": fundamental_data.get("industry", "Unknown"),
                "country": "United States"
            },
            "technical_analysis": technical_indicators,
            "fundamental_analysis": fundamental_data,
            "ai_insights": ai_insights,
            "raw_data": data,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_ai_insights(self, technical: Dict, fundamental: Dict) -> Dict:
        """生成AI洞察"""
        insights = {
            "signals": [],
            "recommendation": "",
            "confidence": 0.5,
            "risk_level": "Medium",
            "target_price": None,
            "stop_loss": None
        }
        
        # 技术面信号
        rsi_14 = technical.get('RSI_14', 50)
        if rsi_14 < 30:
            insights["signals"].append("🟢 RSI超卖，潜在买入机会")
        elif rsi_14 > 70:
            insights["signals"].append("🔴 RSI超买，谨慎风险")
        
        # 移动平均趋势
        sma_20 = technical.get('SMA_20')
        sma_50 = technical.get('SMA_50')
        current_price = technical.get('Current_Price')
        
        if sma_20 and sma_50 and current_price:
            if current_price > sma_20 > sma_50:
                insights["signals"].append("🟢 短期上升趋势确立")
            elif current_price < sma_20 < sma_50:
                insights["signals"].append("🔴 短期下降趋势")
        
        # MACD信号
        macd = technical.get('MACD')
        macd_signal = technical.get('MACD_Signal')
        if macd and macd_signal:
            if macd > macd_signal:
                insights["signals"].append("🟢 MACD金叉，动能向上")
            else:
                insights["signals"].append("🔴 MACD死叉，动能向下")
        
        # 基本面信号
        pe_ratio = fundamental.get('pe_ratio')
        if pe_ratio:
            if pe_ratio < 15:
                insights["signals"].append("🟢 PE估值偏低，价值被低估")
            elif pe_ratio > 30:
                insights["signals"].append("🔴 PE估值偏高，泡沫风险")
        
        # 综合评分
        score = 0
        total_factors = 0
        
        # RSI评分
        if 30 <= rsi_14 <= 70:
            score += 1
        total_factors += 1
        
        # 趋势评分
        if sma_20 and sma_50 and current_price and current_price > sma_20 > sma_50:
            score += 1
        total_factors += 1
        
        # PE评分
        if pe_ratio and 10 <= pe_ratio <= 25:
            score += 1
        total_factors += 1
        
        # ROE评分
        roe = fundamental.get('roe')
        if roe and roe > 0.15:
            score += 1
        total_factors += 1
        
        # 计算置信度
        insights["confidence"] = score / total_factors if total_factors > 0 else 0.5
        
        # 生成建议
        if insights["confidence"] >= 0.75:
            insights["recommendation"] = "🟢 强烈买入"
            insights["risk_level"] = "Low"
        elif insights["confidence"] >= 0.6:
            insights["recommendation"] = "🟡 买入"
            insights["risk_level"] = "Medium"
        elif insights["confidence"] >= 0.4:
            insights["recommendation"] = "🟡 持有"
            insights["risk_level"] = "Medium"
        else:
            insights["recommendation"] = "🔴 观望/卖出"
            insights["risk_level"] = "High"
        
        # 计算目标价和止损
        if current_price:
            atr = technical.get('ATR', 0)
            if atr:
                insights["target_price"] = current_price * (1 + insights["confidence"] * 0.2)
                insights["stop_loss"] = current_price - (atr * 2)
        
        return insights
    
    def process_command(self, text: str) -> Dict:
        """处理命令"""
        parsed = self.command_parser.parse_command(text)
        
        if parsed['action'] == 'analyze':
            return self.comprehensive_analysis(parsed['symbol'])
        elif parsed['action'] == 'compare':
            return self.compare_stocks(parsed['symbols'])
        elif parsed['action'] == 'backtest':
            return self.run_backtest(parsed['symbol'], parsed.get('strategy', 'sma_crossover'))
        elif parsed['action'] == 'screen':
            return self.screen_stocks(parsed)
        else:
            return {
                "error": "抱歉，我没有理解您的指令",
                "examples": [
                    "分析 AAPL / analyze AAPL",
                    "比较 AAPL 和 GOOGL / compare AAPL vs GOOGL", 
                    "回测 TSLA RSI策略 / backtest TSLA RSI strategy",
                    "筛选 PE < 20 / screen PE < 20",
                    "找高分红股票 / find dividend stocks"
                ]
            }
    
    def compare_stocks(self, symbols: List[str]) -> Dict:
        """比较股票"""
        results = {}
        for symbol in symbols:
            analysis = self.comprehensive_analysis(symbol)
            if "error" not in analysis:
                results[symbol] = analysis
        
        if len(results) < 2:
            return {"error": "无法获取足够的股票数据进行比较"}
        
        return {
            "symbols": symbols,
            "analyses": results,
            "comparison_summary": self._generate_comparison_summary(results)
        }
    
    def _generate_comparison_summary(self, results: Dict) -> str:
        """生成比较摘要"""
        summaries = []
        symbols = list(results.keys())
        
        if len(symbols) >= 2:
            stock1, stock2 = symbols[0], symbols[1]
            data1, data2 = results[stock1], results[stock2]
            
            # 价格比较
            price1 = data1['technical_analysis'].get('Current_Price', 0)
            price2 = data2['technical_analysis'].get('Current_Price', 0)
            
            # AI评分比较
            conf1 = data1['ai_insights'].get('confidence', 0)
            conf2 = data2['ai_insights'].get('confidence', 0)
            
            if conf1 > conf2:
                summaries.append(f"{stock1} AI评分更高 ({conf1:.1%} vs {conf2:.1%})")
            else:
                summaries.append(f"{stock2} AI评分更高 ({conf2:.1%} vs {conf1:.1%})")
            
            # PE比较
            pe1 = data1['fundamental_analysis'].get('pe_ratio')
            pe2 = data2['fundamental_analysis'].get('pe_ratio')
            if pe1 and pe2:
                if pe1 < pe2:
                    summaries.append(f"{stock1} PE更低，估值更合理 ({pe1:.1f} vs {pe2:.1f})")
                else:
                    summaries.append(f"{stock2} PE更低，估值更合理 ({pe2:.1f} vs {pe1:.1f})")
        
        return " | ".join(summaries) if summaries else "比较数据不足"
    
    def run_backtest(self, symbol: str, strategy: str) -> Dict:
        """运行回测"""
        data = self.data_manager.get_stock_data(symbol, "2y")
        if data.empty:
            return {"error": f"无法获取 {symbol} 的历史数据"}
        
        # 简化回测逻辑
        if strategy == "sma_crossover":
            return self._backtest_sma_strategy(symbol, data)
        elif strategy == "rsi":
            return self._backtest_rsi_strategy(symbol, data)
        else:
            return {"error": f"不支持的策略: {strategy}"}
    
    def _backtest_sma_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """SMA交叉策略回测"""
        close = data['Close']
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        
        # 生成交易信号
        signals = (sma_20 > sma_50).astype(int)
        returns = close.pct_change()
        strategy_returns = signals.shift(1) * returns
        
        total_return = (1 + strategy_returns).prod() - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        return {
            "symbol": symbol,
            "strategy": "SMA交叉策略",
            "total_return": total_return,
            "annual_volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": self._calculate_max_drawdown(strategy_returns)
        }
    
    def _backtest_rsi_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """RSI策略回测"""
        close = data['Close']
        rsi = self.technical_analyzer._calculate_rsi(close, 14)
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < 30] = 1
        signals[rsi > 70] = 0
        
        returns = close.pct_change()
        strategy_returns = signals.shift(1) * returns
        
        total_return = (1 + strategy_returns).prod() - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        return {
            "symbol": symbol,
            "strategy": "RSI策略",
            "total_return": total_return,
            "annual_volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": self._calculate_max_drawdown(strategy_returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def screen_stocks(self, criteria: Dict) -> Dict:
        """股票筛选"""
        # 扩展的美股池
        us_stock_pool = [
            # FAANG + 科技巨头
            "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE",
            
            # 大盘蓝筹
            "BRK-B", "UNH", "JNJ", "V", "JPM", "PG", "HD", "MA", "BAC", "ABBV",
            "PFE", "KO", "PEP", "MRK", "COST", "WMT", "DIS", "VZ", "CMCSA", "CRM",
            
            # 金融
            "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI", "USB", "TFC", "PNC",
            
            # 工业
            "BA", "CAT", "HON", "UPS", "GE", "LMT", "MMM", "RTX", "DE", "FDX",
            
            # 医疗健康
            "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "MDT", "BMY",
            
            # 消费品
            "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
            
            # 能源
            "XOM", "CVX", "COP", "EOG", "SLB", "KMI", "OXY", "VLO", "PSX", "MPC",
            
            # 房地产
            "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "EXR", "AVB", "EQR", "SPG",
            
            # 公用事业
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ES",
            
            # 通信
            "T", "VZ", "TMUS", "CHTR", "CMCSA", "S", "DISH", "LUMN", "SIRI", "MSGS",
            
            # 热门股票
            "AMD", "INTC", "CRM", "ORCL", "IBM", "QCOM", "TXN", "NOW", "INTU", "MU",
            "PYPL", "SQ", "ROKU", "ZM", "DOCU", "SNAP", "TWTR", "PINS", "UBER", "LYFT",
            
            # ETF
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "GLD", "SLV", "TLT"
        ]
        
        results = []
        for symbol in us_stock_pool:
            try:
                analysis = self.comprehensive_analysis(symbol)
                if "error" not in analysis:
                    if self._meets_screening_criteria(analysis, criteria):
                        results.append({
                            "symbol": symbol,
                            "name": analysis["company_info"]["name"],
                            "sector": analysis["company_info"]["sector"],
                            "price": analysis["technical_analysis"].get("Current_Price"),
                            "pe_ratio": analysis["fundamental_analysis"].get("pe_ratio"),
                            "roe": analysis["fundamental_analysis"].get("roe"),
                            "dividend_yield": analysis["fundamental_analysis"].get("dividend_yield"),
                            "market_cap": analysis["fundamental_analysis"].get("market_cap"),
                            "ai_score": analysis["ai_insights"]["confidence"],
                            "recommendation": analysis["ai_insights"]["recommendation"]
                        })
            except:
                continue
        
        # 排序结果
        if criteria.get('type') == 'dividend':
            results = sorted(results, key=lambda x: x.get('dividend_yield', 0) or 0, reverse=True)
        elif criteria.get('type') == 'growth':
            results = sorted(results, key=lambda x: x.get('ai_score', 0), reverse=True)
        elif criteria.get('type') == 'value':
            results = sorted(results, key=lambda x: x.get('pe_ratio', 999) or 999)
        else:
            results = sorted(results, key=lambda x: x.get('ai_score', 0), reverse=True)
        
        return {
            "criteria": criteria,
            "results": results,
            "count": len(results)
        }
    
    def _meets_screening_criteria(self, analysis: Dict, criteria: Dict) -> bool:
        """检查是否满足筛选条件"""
        fundamental = analysis.get("fundamental_analysis", {})
        technical = analysis.get("technical_analysis", {})
        
        if criteria.get('type') == 'dividend':
            dividend_yield = fundamental.get('dividend_yield', 0)
            return dividend_yield and dividend_yield > 0.03
        elif criteria.get('type') == 'growth':
            revenue_growth = fundamental.get('revenue_growth', 0)
            ai_score = analysis.get("ai_insights", {}).get("confidence", 0)
            return (revenue_growth and revenue_growth > 0.15) or ai_score > 0.7
        elif criteria.get('type') == 'value':
            pe = fundamental.get('pe_ratio', 0)
            return pe and pe < 20 and pe > 0
        elif criteria.get('type') == 'custom':
            # 自定义条件筛选逻辑
            indicator = criteria.get('indicator', '').upper()
            operator = criteria.get('operator', '>')
            value = criteria.get('value', 0)
            
            if indicator in ['PE', 'P/E']:
                pe = fundamental.get('pe_ratio', 0)
                if pe:
                    if operator == '<': return pe < value
                    elif operator == '>': return pe > value
                    elif operator == '=': return abs(pe - value) < 0.1
            elif indicator in ['PB', 'P/B']:
                pb = fundamental.get('pb_ratio', 0)
                if pb:
                    if operator == '<': return pb < value
                    elif operator == '>': return pb > value
                    elif operator == '=': return abs(pb - value) < 0.1
            elif indicator == 'ROE':
                roe = fundamental.get('roe', 0)
                if roe:
                    roe_percent = roe * 100
                    if operator == '<': return roe_percent < value
                    elif operator == '>': return roe_percent > value
                    elif operator == '=': return abs(roe_percent - value) < 1
        
        return True

# 初始化系统
@st.cache_resource
def initialize_us_quant_engine():
    """初始化美股量化引擎"""
    return USStockQuantEngine()

def generate_professional_response(result: Dict) -> Dict:
    """生成专业回复"""
    response = {"role": "assistant", "content": ""}
    
    if "error" in result:
        response["content"] = f"""
## ❌ 系统提示

{result['error']}

### 📚 支持的美股指令格式:
- **股票分析**: `分析 AAPL` / `analyze AAPL`
- **股票对比**: `比较 AAPL 和 GOOGL` / `compare AAPL vs GOOGL`
- **策略回测**: `回测 TSLA RSI策略` / `backtest TSLA RSI strategy`
- **股票筛选**: `筛选 PE < 20` / `screen PE < 20`

### 🇺🇸 支持的美股代码示例:
- **科技股**: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA
- **金融股**: JPM, BAC, WFC, GS, V, MA
- **消费股**: KO, PEP, WMT, HD, MCD, NKE
- **医疗股**: JNJ, PFE, UNH, ABBV, MRK
- **ETF**: SPY, QQQ, IWM, VTI, VOO
"""
        return response
    
    # 全面分析结果
    if "technical_analysis" in result:
        symbol = result["symbol"]
        company_info = result["company_info"]
        technical = result["technical_analysis"]
        fundamental = result["fundamental_analysis"]
        ai_insights = result["ai_insights"]
        
        response["content"] = f"""
## 📈 {symbol} - {company_info['name']} 专业分析报告

### 🏢 公司概况
- **行业**: {company_info['sector']} / {company_info['industry']}
- **市场**: 🇺🇸 美国股票市场
- **当前价格**: ${technical.get('Current_Price', 0):.2f}
- **日涨跌**: {technical.get('Price_Change', 0):+.2f}%

### 🎯 AI智能评估
- **综合评级**: {ai_insights['recommendation']}
- **置信度**: {ai_insights['confidence']:.1%}
- **风险等级**: {ai_insights['risk_level']}
"""
        
        if ai_insights.get('target_price'):
            response["content"] += f"- **目标价位**: ${ai_insights['target_price']:.2f}\n"
        if ai_insights.get('stop_loss'):
            response["content"] += f"- **止损位**: ${ai_insights['stop_loss']:.2f}\n"
        
        # AI信号
        if ai_insights.get('signals'):
            response["content"] += f"""
### 🚨 关键信号
{chr(10).join(f"• {signal}" for signal in ai_insights['signals'])}
"""
        
        # 创建技术指标表格
        tech_data = []
        if technical.get('RSI_14'): tech_data.append(["RSI (14)", f"{technical['RSI_14']:.1f}"])
        if technical.get('SMA_20'): tech_data.append(["SMA 20", f"${technical['SMA_20']:.2f}"])
        if technical.get('SMA_50'): tech_data.append(["SMA 50", f"${technical['SMA_50']:.2f}"])
        if technical.get('MACD'): tech_data.append(["MACD", f"{technical['MACD']:.4f}"])
        if technical.get('BB_Width'): tech_data.append(["布林带宽度", f"{technical['BB_Width']:.2f}%"])
        if technical.get('ATR'): tech_data.append(["ATR", f"{technical['ATR']:.2f}"])
        
        if tech_data:
            tech_df = pd.DataFrame(tech_data, columns=["技术指标", "数值"])
            response["tech_table"] = tech_df
        
        # 创建基本面表格
        fund_data = []
        if fundamental.get('pe_ratio'): fund_data.append(["PE比率", f"{fundamental['pe_ratio']:.2f}"])
        if fundamental.get('pb_ratio'): fund_data.append(["PB比率", f"{fundamental['pb_ratio']:.2f}"])
        if fundamental.get('roe'): fund_data.append(["ROE", f"{fundamental['roe']*100:.1f}%"])
        if fundamental.get('debt_to_equity'): fund_data.append(["债务股权比", f"{fundamental['debt_to_equity']:.2f}"])
        if fundamental.get('current_ratio'): fund_data.append(["流动比率", f"{fundamental['current_ratio']:.2f}"])
        if fundamental.get('dividend_yield'): fund_data.append(["股息率", f"{fundamental['dividend_yield']*100:.2f}%"])
        if fundamental.get('market_cap'): fund_data.append(["市值", f"${fundamental['market_cap']/1e9:.1f}B"])
        
        if fund_data:
            fund_df = pd.DataFrame(fund_data, columns=["基本面指标", "数值"])
            response["fund_table"] = fund_df
        
        # 生成专业图表
        data = result["raw_data"]
        if not data.empty:
            # 创建综合技术分析图表
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('价格与移动平均', 'RSI', 'MACD'),
                vertical_spacing=0.08,
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # 价格线
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], name='收盘价', 
                          line=dict(color='#3b82f6', width=2)),
                row=1, col=1
            )
            
            # 移动平均线
            if len(data) >= 20:
                sma_20 = data['Close'].rolling(20).mean()
                fig.add_trace(
                    go.Scatter(x=data.index, y=sma_20, name='SMA 20',
                              line=dict(color='#f59e0b', width=1)),
                    row=1, col=1
                )
            
            if len(data) >= 50:
                sma_50 = data['Close'].rolling(50).mean()
                fig.add_trace(
                    go.Scatter(x=data.index, y=sma_50, name='SMA 50',
                              line=dict(color='#ef4444', width=1)),
                    row=1, col=1
                )
            
            # RSI
            if len(data) >= 14:
                rsi = AdvancedTechnicalIndicators._calculate_rsi(data['Close'], 14)
                fig.add_trace(
                    go.Scatter(x=data.index, y=rsi, name='RSI',
                              line=dict(color='#8b5cf6', width=2)),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            if len(data) >= 26:
                macd_data = AdvancedTechnicalIndicators._calculate_macd(data['Close'])
                if macd_data['MACD'] is not None:
                    ema_12 = data['Close'].ewm(span=12).mean()
                    ema_26 = data['Close'].ewm(span=26).mean()
                    macd_line = ema_12 - ema_26
                    signal_line = macd_line.ewm(span=9).mean()
                    
                    fig.add_trace(
                        go.Scatter(x=data.index, y=macd_line, name='MACD',
                                  line=dict(color='#06b6d4', width=2)),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=data.index, y=signal_line, name='Signal',
                                  line=dict(color='#f97316', width=1)),
                        row=3, col=1
                    )
            
            fig.update_layout(
                title=f"{symbol} 专业技术分析图表",
                height=800,
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            response["chart_data"] = fig
    
    # 比较分析结果
    elif "comparison_summary" in result:
        symbols = result["symbols"]
        analyses = result["analyses"]
        
        response["content"] = f"""
## ⚖️ {' vs '.join(symbols)} 美股对比分析

### 📊 对比摘要
{result['comparison_summary']}

### 📋 详细对比数据
"""
        
        # 创建对比表格
        comparison_data = []
        for symbol, analysis in analyses.items():
            comparison_data.append({
                "股票代码": symbol,
                "公司名称": analysis["company_info"]["name"][:20],
                "行业": analysis["company_info"]["sector"][:15],
                "当前价格": f"${analysis['technical_analysis'].get('Current_Price', 0):.2f}",
                "PE比率": f"{analysis['fundamental_analysis'].get('pe_ratio', 0):.1f}" if analysis['fundamental_analysis'].get('pe_ratio') else 'N/A',
                "ROE": f"{(analysis['fundamental_analysis'].get('roe', 0) or 0)*100:.1f}%" if analysis['fundamental_analysis'].get('roe') else 'N/A',
                "市值": f"${(analysis['fundamental_analysis'].get('market_cap', 0) or 0)/1e9:.1f}B" if analysis['fundamental_analysis'].get('market_cap') else 'N/A',
                "AI评分": f"{analysis['ai_insights']['confidence']:.1%}",
                "推荐": analysis['ai_insights']['recommendation']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        response["table_data"] = comparison_df
    
    # 回测结果
    elif "strategy" in result:
        symbol = result['symbol']
        strategy = result['strategy']
        
        response["content"] = f"""
## 🔬 {symbol} - {strategy} 回测报告

### 📈 绩效指标
- **总收益率**: {result['total_return']:.2%}
- **年化波动率**: {result['annual_volatility']:.2%}
- **夏普比率**: {result['sharpe_ratio']:.3f}
- **最大回撤**: {result['max_drawdown']:.2%}

### 🎯 策略评估
"""
        
        sharpe = result['sharpe_ratio']
        if sharpe > 1.5:
            response["content"] += "🟢 **卓越策略** - 夏普比率>1.5，风险调整收益优异"
        elif sharpe > 1.0:
            response["content"] += "🟢 **优秀策略** - 夏普比率>1.0，表现良好"
        elif sharpe > 0.5:
            response["content"] += "🟡 **可接受策略** - 夏普比率>0.5，有一定价值"
        else:
            response["content"] += "🔴 **不推荐策略** - 夏普比率<0.5，风险过高"
    
    # 筛选结果
    elif "results" in result:
        criteria = result['criteria']
        results = result['results']
        count = result['count']
        
        response["content"] = f"""
## 🔍 美股专业筛选报告

### 📊 筛选条件
"""
        
        if criteria.get('type') == 'dividend':
            response["content"] += "- **筛选类型**: 高分红美股 (股息率 > 3%)"
        elif criteria.get('type') == 'growth':
            response["content"] += "- **筛选类型**: 成长型美股 (高AI评分或营收增长 > 15%)"
        elif criteria.get('type') == 'value':
            response["content"] += "- **筛选类型**: 价值型美股 (PE < 20)"
        elif criteria.get('type') == 'custom':
            indicator = criteria.get('indicator', '')
            operator = criteria.get('operator', '')
            value = criteria.get('value', 0)
            response["content"] += f"- **筛选类型**: 自定义条件 ({indicator} {operator} {value})"
        
        response["content"] += f"""

### 📋 筛选结果 (共发现 {count} 只美股)
"""
        
        if results:
            # 创建结果表格
            results_data = []
            for i, stock in enumerate(results[:20]):  # 显示前20只
                results_data.append({
                    "排名": i + 1,
                    "代码": stock['symbol'],
                    "公司名称": stock.get('name', stock['symbol'])[:25],
                    "行业": stock.get('sector', 'N/A')[:15],
                    "价格": f"${stock.get('price', 0):.2f}" if stock.get('price') else 'N/A',
                    "PE": f"{stock.get('pe_ratio', 0):.1f}" if stock.get('pe_ratio') else 'N/A',
                    "ROE": f"{(stock.get('roe', 0) or 0)*100:.1f}%" if stock.get('roe') else 'N/A',
                    "股息率": f"{(stock.get('dividend_yield', 0) or 0)*100:.2f}%" if stock.get('dividend_yield') else 'N/A',
                    "市值": f"${(stock.get('market_cap', 0) or 0)/1e9:.1f}B" if stock.get('market_cap') else 'N/A',
                    "AI评分": f"{(stock.get('ai_score', 0) or 0):.1%}",
                    "推荐": stock.get('recommendation', 'N/A')[:8]
                })
            
            results_df = pd.DataFrame(results_data)
            response["table_data"] = results_df
            
            if count > 20:
                response["content"] += f"\n*显示前20只股票，总共筛选出{count}只符合条件的美股*"
        else:
            response["content"] += "\n❌ 未找到符合条件的美股，建议调整筛选条件"
    
    return response

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "us_quant_engine" not in st.session_state:
    st.session_state.us_quant_engine = initialize_us_quant_engine()

# 专业头部
st.markdown("""
<div class="pro-header">
    <h1 class="terminal-title">🇺🇸 QUANTGPT PRO</h1>
    <p class="terminal-subtitle">Professional US Stock AI Trading Terminal | 专业美股AI交易终端</p>
</div>
""", unsafe_allow_html=True)

# 状态栏
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="status-indicator">🟢 SYSTEM ONLINE</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="status-indicator">🇺🇸 US MARKETS</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="status-indicator">🤖 AI ENGINE ACTIVE</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="status-indicator">📊 REAL-TIME DATA</div>', unsafe_allow_html=True)

# 侧边栏专业配置
with st.sidebar:
    st.markdown("### 🎛️ 美股交易终端设置")
    
    # 分析周期
    analysis_period = st.selectbox(
        "📅 分析周期",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )
    
    # 风险偏好
    risk_preference = st.selectbox(
        "⚡ 风险偏好",
        ["保守型", "平衡型", "成长型", "激进型"],
        index=1
    )
    
    # 投资风格
    investment_style = st.selectbox(
        "💎 投资风格",
        ["价值投资", "成长投资", "动量投资", "分红投资"],
        index=0
    )
    
    st.markdown("---")
    
    # 实时美股指数
    st.markdown("### 📈 美股指数实时概览")
    
    # 主要美股指数
    us_indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC", 
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT"
    }
    
    for name, symbol in us_indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = (current - prev) / prev * 100
                color = "#10b981" if change > 0 else "#ef4444"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {color}">{change:+.2f}%</div>
                    <div class="metric-label">{name}</div>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">--</div>
                <div class="metric-label">{name}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 美股功能指南
    st.markdown("### 📚 美股功能指南")
    
    with st.expander("🔍 股票分析", expanded=True):
        st.markdown("""
        **支持格式:**
        - `分析 AAPL` / `analyze AAPL`
        - `查看 GOOGL` / `check GOOGL`
        - `看看 TSLA` / `look at TSLA`
        
        **热门美股:**
        - 🍎 科技: AAPL, GOOGL, MSFT, AMZN, TSLA
        - 💰 金融: JPM, BAC, V, MA, BRK-B
        - 🛍️ 消费: WMT, HD, KO, PEP, MCD
        - 💊 医疗: JNJ, PFE, UNH, ABBV
        """)
    
    with st.expander("📊 技术指标", expanded=False):
        st.markdown("""
        **包含指标:**
        - 移动平均线 (SMA/EMA 5-200日)
        - RSI、MACD、KDJ
        - 布林带、威廉指标
        - ATR波动率、CCI
        - 成交量分析
        """)
    
    with st.expander("💎 基本面分析", expanded=False):
        st.markdown("""
        **财务指标:**
        - 估值: PE, PB, PS, PEG
        - 盈利: ROE, ROA, 毛利率
        - 财务: 债务比率, 流动比率
        - 成长: 营收增长, 盈利增长
        - 股息: 股息率, 分红比率
        """)
    
    with st.expander("🔬 策略回测", expanded=False):
        st.markdown("""
        **支持策略:**
        - SMA交叉策略
        - RSI超买超卖策略
        - MACD金叉死叉策略
        
        **回测指标:**
        - 总收益率、年化波动率
        - 夏普比率、最大回撤
        """)

# 示例命令 (如果没有历史消息)
if not st.session_state.messages:
    st.markdown("### 🚀 美股快速开始")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 分析 AAPL", key="ex1", help="苹果公司深度分析"):
            st.session_state.messages.append({"role": "user", "content": "分析 AAPL"})
            st.rerun()
    
    with col2:
        if st.button("🔍 筛选价值股", key="ex2", help="筛选低估值美股"):
            st.session_state.messages.append({"role": "user", "content": "找价值股票"})
            st.rerun()
    
    with col3:
        if st.button("⚖️ 对比 AAPL vs GOOGL", key="ex3", help="科技巨头对比"):
            st.session_state.messages.append({"role": "user", "content": "比较 AAPL 和 GOOGL"})
            st.rerun()
    
    with col4:
        if st.button("🔬 回测 TSLA RSI", key="ex4", help="特斯拉RSI策略"):
            st.session_state.messages.append({"role": "user", "content": "回测 TSLA RSI策略"})
            st.rerun()
    
    # 高级美股示例
    st.markdown("### 🎯 高级美股分析")
    
    advanced_col1, advanced_col2 = st.columns(2)
    
    with advanced_col1:
        if st.button("💰 分析伯克希尔 BRK-B", key="ex5"):
            st.session_state.messages.append({"role": "user", "content": "分析 BRK-B"})
            st.rerun()
        
        if st.button("💎 筛选 PE < 15 的价值股", key="ex6"):
            st.session_state.messages.append({"role": "user", "content": "筛选 PE < 15"})
            st.rerun()
        
        if st.button("🏦 分析摩根大通 JPM", key="ex7"):
            st.session_state.messages.append({"role": "user", "content": "分析 JPM"})
            st.rerun()
    
    with advanced_col2:
        if st.button("🚀 找高成长科技股", key="ex8"):
            st.session_state.messages.append({"role": "user", "content": "找成长股票"})
            st.rerun()
        
        if st.button("💰 找高分红蓝筹股", key="ex9"):
            st.session_state.messages.append({"role": "user", "content": "找高分红股票"})
            st.rerun()
        
        if st.button("📈 对比 SPY vs QQQ", key="ex10"):
            st.session_state.messages.append({"role": "user", "content": "比较 SPY 和 QQQ"})
            st.rerun()

# 显示聊天历史
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 TRADER:</strong><br/>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-message">
            <strong>🤖 QUANTGPT PRO:</strong><br/>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示内容
        st.markdown(message["content"])
        
        # 显示图表
        if "chart_data" in message:
            st.plotly_chart(message["chart_data"], use_container_width=True, config={'displayModeBar': False})
        
        # 显示表格
        if "table_data" in message:
            st.dataframe(
                message["table_data"], 
                use_container_width=True,
                hide_index=True
            )
        
        if "tech_table" in message:
            st.markdown("### 📊 技术指标详情")
            st.dataframe(
                message["tech_table"], 
                use_container_width=True,
                hide_index=True
            )
        
        if "fund_table" in message:
            st.markdown("### 💎 基本面指标详情")
            st.dataframe(
                message["fund_table"], 
                use_container_width=True,
                hide_index=True
            )

# 专业输入区域
st.markdown("### 💬 美股交易指令终端")

# 创建输入框
user_input = st.text_input(
    "美股交易指令",
    placeholder="输入美股指令: 分析 AAPL | analyze GOOGL | 比较 AAPL vs TSLA | 筛选 PE < 20 | 找高分红股票...",
    key="user_input",
    label_visibility="collapsed"
)

# 发送按钮
col1, col2, col3 = st.columns([5, 1, 1])
with col2:
    send_button = st.button("🚀 EXECUTE", type="primary", use_container_width=True)
with col3:
    if st.session_state.messages:
        clear_button = st.button("🗑️ CLEAR", use_container_width=True)
        if clear_button:
            st.session_state.messages = []
            st.rerun()

# 处理用户输入
if send_button and user_input.strip():
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 显示处理状态
    with st.spinner("🤖 QuantGPT Pro 正在进行美股专业分析..."):
        try:
            # 处理命令
            result = st.session_state.us_quant_engine.process_command(user_input)
            
            # 生成AI回复
            ai_response = generate_professional_response(result)
            
            # 添加AI消息
            st.session_state.messages.append(ai_response)
            
        except Exception as e:
            error_response = {
                "role": "assistant",
                "content": f"""
## ❌ 系统错误

处理美股指令时发生错误: {str(e)}

### 🔧 故障排除建议:
- 检查美股代码格式是否正确 (如: AAPL, GOOGL, TSLA)
- 确认网络连接状态
- 稍后重试或联系技术支持

### 📞 技术支持:
如问题持续，请报告错误详情以获取帮助。
"""
            }
            st.session_state.messages.append(error_response)
    
    # 刷新页面
    st.rerun()

# 专业页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem; font-family: "JetBrains Mono", monospace;'>
    <div style='margin-bottom: 1rem;'>
        <span class="pro-badge">QUANTGPT PRO US v3.0</span>
    </div>
    <p><strong>🇺🇸 Professional US Stock AI Trading Terminal</strong></p>
    <p>⚡ Real-time US Market Analysis | 🔬 Advanced Technical Indicators | 💎 Comprehensive Fundamental Analysis</p>
    <p><small>⚠️ Professional US stock trading tool for reference only. Investment involves risks.</small></p>
    <p><small>📊 Supports: NYSE, NASDAQ, AMEX | 🕐 Market Hours: 9:30 AM - 4:00 PM ET</small></p>
</div>
""", unsafe_allow_html=True)
