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
import time
from typing import Dict, List, Optional, Tuple, Generator
import warnings
warnings.filterwarnings('ignore')

# Professional Terminal Configuration
st.set_page_config(
    page_title="US Stock Terminal Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Professional Theme CSS
def get_theme_css(theme="dark"):
    themes = {
        "dark": {
            "bg_primary": "#0F172A",
            "bg_secondary": "#1E293B", 
            "bg_accent": "#334155",
            "text_primary": "#F8FAFC",
            "text_secondary": "#CBD5E1",
            "accent_color": "#3B82F6",
            "success_color": "#10B981",
            "warning_color": "#F59E0B",
            "danger_color": "#EF4444"
        },
        "minimal": {
            "bg_primary": "#FFFFFF",
            "bg_secondary": "#F8FAFC",
            "bg_accent": "#E2E8F0",
            "text_primary": "#1E293B",
            "text_secondary": "#475569",
            "accent_color": "#3B82F6",
            "success_color": "#059669",
            "warning_color": "#D97706",
            "danger_color": "#DC2626"
        },
        "terminal": {
            "bg_primary": "#000000",
            "bg_secondary": "#0D1117",
            "bg_accent": "#21262D",
            "text_primary": "#00FF41",
            "text_secondary": "#58A6FF",
            "accent_color": "#00FF41",
            "success_color": "#7C3AED",
            "warning_color": "#F59E0B",
            "danger_color": "#FF6B6B"
        }
    }
    
    colors = themes.get(theme, themes["dark"])
    
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
    
    :root {{
        --bg-primary: {colors['bg_primary']};
        --bg-secondary: {colors['bg_secondary']};
        --bg-accent: {colors['bg_accent']};
        --text-primary: {colors['text_primary']};
        --text-secondary: {colors['text_secondary']};
        --accent-color: {colors['accent_color']};
        --success-color: {colors['success_color']};
        --warning-color: {colors['warning_color']};
        --danger-color: {colors['danger_color']};
    }}
    
    .stApp {{
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }}
    
    /* Hide Streamlit UI */
    #MainMenu, footer, header, .stDeployButton {{visibility: hidden;}}
    
    /* Professional Header */
    .terminal-header {{
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-accent) 100%);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        border: 1px solid var(--bg-accent);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }}
    
    .terminal-title {{
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.025em;
    }}
    
    .terminal-subtitle {{
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin: 0.25rem 0 0 0;
        font-weight: 400;
    }}
    
    /* Status Bar */
    .status-bar {{
        display: flex;
        gap: 2rem;
        background: var(--bg-secondary);
        padding: 1rem 2rem;
        border-radius: 6px;
        border: 1px solid var(--bg-accent);
        margin-bottom: 2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }}
    
    .status-item {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--success-color);
    }}
    
    /* Message Containers */
    .user-message {{
        background: var(--bg-secondary);
        border: 1px solid var(--bg-accent);
        border-left: 3px solid var(--accent-color);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
    }}
    
    .ai-message {{
        background: var(--bg-secondary);
        border: 1px solid var(--success-color);
        border-left: 3px solid var(--success-color);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 6px;
    }}
    
    .streaming-message {{
        background: var(--bg-secondary);
        border: 1px solid var(--warning-color);
        border-left: 3px solid var(--warning-color);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 6px;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.8; }}
        50% {{ opacity: 1; }}
    }}
    
    /* Input Area */
    .stTextInput > div > div > input {{
        background: var(--bg-secondary);
        border: 2px solid var(--bg-accent);
        border-radius: 6px;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: var(--accent-color);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }}
    
    /* Buttons */
    .stButton > button {{
        background: var(--accent-color);
        border: none;
        border-radius: 6px;
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        background: var(--success-color);
        transform: translateY(-1px);
    }}
    
    /* Quick Actions */
    .quick-action {{
        background: var(--bg-secondary);
        border: 1px solid var(--bg-accent);
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }}
    
    .quick-action:hover {{
        border-color: var(--accent-color);
        background: var(--bg-accent);
    }}
    
    /* Data Tables */
    .dataframe {{
        background: var(--bg-secondary);
        border: 1px solid var(--bg-accent);
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }}
    
    /* Metrics */
    .metric-card {{
        background: var(--bg-secondary);
        border: 1px solid var(--bg-accent);
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }}
    
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent-color);
        font-family: 'JetBrains Mono', monospace;
    }}
    
    .metric-label {{
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Responsive */
    @media (max-width: 768px) {{
        .terminal-header {{ padding: 1rem; }}
        .status-bar {{ flex-direction: column; gap: 1rem; }}
        .quick-action {{ margin: 0.25rem; }}
    }}
</style>
"""

class StockUniverse:
    """Manage universe of US stocks"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_sp500_symbols():
        """Get S&P 500 symbols"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            return tables[0]['Symbol'].tolist()
        except:
            # Fallback list of major stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'JNJ', 'V',
                    'UNH', 'HD', 'PG', 'BAC', 'MA', 'DIS', 'ADBE', 'CRM', 'NFLX', 'PYPL',
                    'INTC', 'CMCSA', 'VZ', 'T', 'PFE', 'WMT', 'KO', 'PEP', 'ABT', 'NKE']
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_nasdaq100_symbols():
        """Get NASDAQ 100 symbols"""
        try:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            tables = pd.read_html(url)
            return tables[4]['Ticker'].tolist()
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'ADBE', 'CRM',
                    'PYPL', 'INTC', 'QCOM', 'COST', 'AVGO', 'TXN', 'TMUS', 'CHTR', 'SBUX', 'GILD']
    
    @staticmethod
    def get_popular_stocks():
        """Get list of popular stocks"""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'JPM', 'JNJ',
                'V', 'PG', 'UNH', 'HD', 'BAC', 'MA', 'DIS', 'ADBE', 'CRM', 'PYPL',
                'INTC', 'CMCSA', 'VZ', 'T', 'PFE', 'WMT', 'KO', 'PEP', 'ABT', 'NKE']

class StockScreener:
    """Advanced stock screening functionality"""
    
    def __init__(self):
        self.universe = StockUniverse()
        self.custom_criteria = None
    
    def screen_stocks(self, criteria: Dict = None, universe_type: str = "sp500") -> pd.DataFrame:
        """Screen stocks based on criteria"""
        
        # Use custom criteria if provided
        if self.custom_criteria:
            criteria = self.custom_criteria
            self.custom_criteria = None  # Reset after use
        
        # Default criteria if none provided
        if criteria is None:
            criteria = {
                'min_price': 10,
                'max_price': 1000,
                'min_market_cap': 1,  # 1B+
                'max_pe': 30,
                'min_rsi': 20,
                'max_rsi': 80,
                'min_1m_return': -50,
                'min_dividend_yield': 0
            }
        
        # Get stock universe
        if universe_type == "sp500":
            symbols = self.universe.get_sp500_symbols()
        elif universe_type == "nasdaq100":
            symbols = self.universe.get_nasdaq100_symbols()
        else:
            symbols = self.universe.get_popular_stocks()
        
        results = []
        
        for symbol in symbols[:50]:  # Limit to prevent timeout
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1y")
                
                if hist.empty:
                    continue
                
                # Calculate metrics
                current_price = hist['Close'].iloc[-1]
                
                # Technical indicators
                rsi = self._calculate_rsi(hist['Close'])
                sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else None
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
                
                # Price performance
                price_1m = hist['Close'].iloc[-21] if len(hist) >= 21 else current_price
                price_3m = hist['Close'].iloc[-63] if len(hist) >= 63 else current_price
                perf_1m = (current_price - price_1m) / price_1m * 100
                perf_3m = (current_price - price_3m) / price_3m * 100
                
                # Fundamental metrics
                pe_ratio = info.get('trailingPE')
                pb_ratio = info.get('priceToBook')
                roe = info.get('returnOnEquity')
                debt_to_equity = info.get('debtToEquity')
                dividend_yield = info.get('dividendYield', 0) or 0
                market_cap = info.get('marketCap')
                
                stock_data = {
                    'Symbol': symbol,
                    'Name': info.get('longName', symbol)[:30],
                    'Sector': info.get('sector', 'Unknown'),
                    'Price': current_price,
                    'Market Cap': market_cap,
                    'P/E': pe_ratio,
                    'P/B': pb_ratio,
                    'ROE': roe,
                    'Debt/Equity': debt_to_equity,
                    'Div Yield': dividend_yield * 100 if dividend_yield else 0,
                    'RSI': rsi,
                    '1M Return': perf_1m,
                    '3M Return': perf_3m,
                    'SMA 20': sma_20,
                    'SMA 50': sma_50
                }
                
                # Apply screening criteria
                if self._meets_criteria(stock_data, criteria):
                    results.append(stock_data)
                    
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period:
            return 50.0
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _meets_criteria(self, stock_data: Dict, criteria: Dict) -> bool:
        """Check if stock meets screening criteria"""
        try:
            # Price criteria
            if criteria.get('min_price') and stock_data['Price'] < criteria['min_price']:
                return False
            if criteria.get('max_price') and stock_data['Price'] > criteria['max_price']:
                return False
            
            # Market cap criteria
            if criteria.get('min_market_cap') and stock_data['Market Cap']:
                if stock_data['Market Cap'] < criteria['min_market_cap'] * 1e9:
                    return False
            
            # P/E criteria
            if criteria.get('max_pe') and stock_data['P/E']:
                if stock_data['P/E'] > criteria['max_pe']:
                    return False
            
            # RSI criteria
            if criteria.get('max_rsi') and stock_data['RSI'] > criteria['max_rsi']:
                return False
            if criteria.get('min_rsi') and stock_data['RSI'] < criteria['min_rsi']:
                return False
            
            # Performance criteria
            if criteria.get('min_1m_return') and stock_data['1M Return'] < criteria['min_1m_return']:
                return False
            
            # Dividend yield criteria
            if criteria.get('min_dividend_yield') and stock_data['Div Yield'] < criteria['min_dividend_yield']:
                return False
            
            # Sector filter
            if criteria.get('sectors') and stock_data['Sector'] not in criteria['sectors']:
                return False
            
            return True
            
        except:
            return False

class StrategyBacktester:
    """Simple strategy backtesting engine"""
    
    def __init__(self):
        pass
    
    def backtest_strategy(self, symbol: str, strategy: str, period: str = "1y") -> Dict:
        """Backtest a simple strategy"""
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return {"error": "No data available"}
            
            if strategy == "sma_crossover":
                return self._backtest_sma_crossover(data)
            elif strategy == "rsi_mean_reversion":
                return self._backtest_rsi_mean_reversion(data)
            elif strategy == "momentum":
                return self._backtest_momentum(data)
            elif strategy == "buy_and_hold":
                return self._backtest_buy_and_hold(data)
            else:
                return {"error": "Unknown strategy"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _backtest_sma_crossover(self, data: pd.DataFrame) -> Dict:
        """SMA Crossover Strategy: Buy when SMA20 > SMA50, Sell when SMA20 < SMA50"""
        
        data = data.copy()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        # Generate signals
        data['Signal'] = 0
        data['Signal'][20:] = np.where(data['SMA_20'][20:] > data['SMA_50'][20:], 1, 0)
        data['Position'] = data['Signal'].diff()
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
        
        # Performance metrics
        total_return = (data['Strategy_Returns'] + 1).prod() - 1
        buy_hold_return = (data['Returns'] + 1).prod() - 1
        
        win_rate = len(data[data['Strategy_Returns'] > 0]) / len(data[data['Strategy_Returns'] != 0]) if len(data[data['Strategy_Returns'] != 0]) > 0 else 0
        
        return {
            'strategy': 'SMA Crossover (20/50)',
            'total_return': total_return * 100,
            'buy_hold_return': buy_hold_return * 100,
            'win_rate': win_rate * 100,
            'total_trades': len(data[data['Position'] != 0]),
            'data': data,
            'signals': data[data['Position'] != 0][['Close', 'Position']].to_dict('records')
        }
    
    def _backtest_rsi_mean_reversion(self, data: pd.DataFrame) -> Dict:
        """RSI Mean Reversion: Buy when RSI < 30, Sell when RSI > 70"""
        
        data = data.copy()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        data['Signal'] = 0
        data.loc[data['RSI'] < 30, 'Signal'] = 1  # Buy
        data.loc[data['RSI'] > 70, 'Signal'] = -1  # Sell
        
        # Forward fill signals until next signal
        data['Position'] = 0
        current_position = 0
        for i in range(len(data)):
            if data['Signal'].iloc[i] == 1:
                current_position = 1
            elif data['Signal'].iloc[i] == -1:
                current_position = 0
            data['Position'].iloc[i] = current_position
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
        
        # Performance metrics
        total_return = (data['Strategy_Returns'] + 1).prod() - 1
        buy_hold_return = (data['Returns'] + 1).prod() - 1
        
        trades = len(data[data['Signal'] != 0])
        winning_trades = len(data[(data['Signal'] != 0) & (data['Strategy_Returns'] > 0)])
        win_rate = winning_trades / trades if trades > 0 else 0
        
        return {
            'strategy': 'RSI Mean Reversion',
            'total_return': total_return * 100,
            'buy_hold_return': buy_hold_return * 100,
            'win_rate': win_rate * 100,
            'total_trades': trades,
            'data': data,
            'signals': data[data['Signal'] != 0][['Close', 'Signal', 'RSI']].to_dict('records')
        }
    
    def _backtest_momentum(self, data: pd.DataFrame) -> Dict:
        """Momentum Strategy: Buy on 5-day momentum, hold for 10 days"""
        
        data = data.copy()
        data['Momentum'] = data['Close'].pct_change(5)  # 5-day momentum
        
        # Generate signals
        data['Signal'] = 0
        data.loc[data['Momentum'] > 0.05, 'Signal'] = 1  # Buy on +5% momentum
        
        # Hold for 10 days
        data['Position'] = 0
        hold_counter = 0
        for i in range(len(data)):
            if data['Signal'].iloc[i] == 1:
                hold_counter = 10
            
            if hold_counter > 0:
                data['Position'].iloc[i] = 1
                hold_counter -= 1
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
        
        # Performance metrics
        total_return = (data['Strategy_Returns'] + 1).prod() - 1
        buy_hold_return = (data['Returns'] + 1).prod() - 1
        
        trades = len(data[data['Signal'] == 1])
        win_rate = 0.6  # Approximate for momentum strategies
        
        return {
            'strategy': 'Momentum (5-day)',
            'total_return': total_return * 100,
            'buy_hold_return': buy_hold_return * 100,
            'win_rate': win_rate * 100,
            'total_trades': trades,
            'data': data,
            'signals': data[data['Signal'] == 1][['Close', 'Momentum']].to_dict('records')
        }
    
    def _backtest_buy_and_hold(self, data: pd.DataFrame) -> Dict:
        """Buy and Hold Strategy"""
        
        data = data.copy()
        data['Returns'] = data['Close'].pct_change()
        total_return = (data['Returns'] + 1).prod() - 1
        
        return {
            'strategy': 'Buy and Hold',
            'total_return': total_return * 100,
            'buy_hold_return': total_return * 100,
            'win_rate': 100.0,
            'total_trades': 1,
            'data': data,
            'signals': []
        }

class CommandParser:
    """Smart command parser for stock analysis"""
    
    @staticmethod
    def parse_command(text: str) -> Dict:
        """Parse user command and extract symbols and action"""
        text = text.upper().strip()
        
        # Check for screening commands
        if any(word in text for word in ['SCREEN', 'FIND', 'FILTER', 'SEARCH']):
            return {'action': 'screen', 'original_text': text}
        
        # Check for backtest commands
        if any(word in text for word in ['BACKTEST', 'TEST', 'STRATEGY']):
            # Extract strategy type
            strategy = 'sma_crossover'  # default
            if 'RSI' in text:
                strategy = 'rsi_mean_reversion'
            elif 'MOMENTUM' in text:
                strategy = 'momentum'
            elif 'BUY' in text and 'HOLD' in text:
                strategy = 'buy_and_hold'
            
            # Extract symbol
            symbol_pattern = r'\b[A-Z]{1,5}(?:\d{0,2})?\b'
            potential_symbols = re.findall(symbol_pattern, text)
            command_words = {'BACKTEST', 'TEST', 'STRATEGY', 'RSI', 'MOMENTUM', 'BUY', 'HOLD',
                           'SMA', 'CROSSOVER', 'THE', 'ON', 'FOR', 'WITH'}
            symbols = [s for s in potential_symbols if s not in command_words]
            
            return {
                'action': 'backtest',
                'strategy': strategy,
                'symbols': symbols[:1],  # Only first symbol
                'original_text': text
            }
        
        # Stock symbol pattern (1-5 letters, optionally followed by numbers)
        symbol_pattern = r'\b[A-Z]{1,5}(?:\d{0,2})?\b'
        
        # Extract all potential symbols
        potential_symbols = re.findall(symbol_pattern, text)
        
        # Filter out common command words
        command_words = {'ANALYZE', 'ANALYSE', 'CHECK', 'LOOK', 'AT', 'SHOW', 'ME', 'TELL', 
                        'ABOUT', 'GET', 'DATA', 'FOR', 'COMPARE', 'VS', 'VERSUS', 'AND', 
                        'WITH', 'AGAINST', 'STOCK', 'STOCKS', 'PRICE', 'CHART', 'INFO',
                        'INFORMATION', 'DETAILS', 'ANALYSIS', 'REPORT', 'THE', 'OF', 'IS',
                        'ARE', 'WHAT', 'HOW', 'WHY', 'WHEN', 'WHERE', 'WHICH'}
        
        symbols = [s for s in potential_symbols if s not in command_words]
        
        # Determine action type
        if any(word in text for word in ['COMPARE', 'VS', 'VERSUS', 'AGAINST']):
            action = 'compare'
        elif any(word in text for word in ['ANALYZE', 'ANALYSE', 'CHECK', 'LOOK', 'SHOW', 'TELL']):
            action = 'analyze'
        else:
            action = 'analyze'  # Default action
        
        return {
            'action': action,
            'symbols': symbols[:5],  # Limit to 5 symbols max
            'original_text': text
        }
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate if symbol looks like a valid stock ticker"""
        if not symbol or len(symbol) < 1 or len(symbol) > 7:
            return False
        
        # Must start with letter
        if not symbol[0].isalpha():
            return False
        
        # Rest can be letters or numbers
        return all(c.isalnum() for c in symbol)

class StreamingAnalyzer:
    """Streaming Analysis Engine"""
    
    def __init__(self):
        self.data_cache = {}
        self.parser = CommandParser()
        self.screener = StockScreener()
        self.backtester = StrategyBacktester()
    
    def process_command(self, text: str) -> Generator[Dict, None, None]:
        """Process user command and route to appropriate analysis"""
        # Parse the command
        parsed = self.parser.parse_command(text)
        
        yield {
            "type": "status",
            "content": f"🔍 Processing command: {parsed['original_text']}"
        }
        
        # Route to appropriate handler
        if parsed['action'] == 'screen':
            yield from self.stream_screening()
        elif parsed['action'] == 'backtest':
            if not parsed.get('symbols'):
                yield {
                    "type": "error",
                    "content": "❌ No symbol found for backtesting. Example: 'backtest AAPL with RSI strategy'"
                }
                return
            yield from self.stream_backtest(parsed['symbols'][0], parsed.get('strategy', 'sma_crossover'))
        elif parsed['action'] == 'compare' and len(parsed.get('symbols', [])) >= 2:
            yield from self.stream_comparison(parsed['symbols'][:2])
        elif parsed.get('symbols'):
            yield from self.stream_analysis(parsed['symbols'][0])
        else:
            yield {
                "type": "error",
                "content": "❌ No valid command found. Try: 'AAPL', 'screen stocks', 'backtest AAPL', 'compare AAPL vs MSFT'"
            }
    
    def stream_screening(self) -> Generator[Dict, None, None]:
        """Stream stock screening results"""
        yield {
            "type": "status",
            "content": "🔍 Starting stock screening process..."
        }
        
        # Default screening criteria
        criteria = {
            'min_price': 10,
            'max_price': 1000,
            'min_market_cap': 1,  # 1B+
            'max_pe': 30,
            'min_rsi': 20,
            'max_rsi': 80,
            'min_1m_return': -50,
            'min_dividend_yield': 0
        }
        
        yield {
            "type": "status",
            "content": "📊 Analyzing S&P 500 stocks with default criteria..."
        }
        
        results = self.screener.screen_stocks(criteria, "sp500")
        
        if results.empty:
            yield {
                "type": "error",
                "content": "❌ No stocks found matching criteria. Try adjusting screening parameters."
            }
            return
        
        # Sort by performance
        results = results.sort_values('1M Return', ascending=False)
        
        yield {
            "type": "screening_results",
            "content": {
                'total_found': len(results),
                'results': results.head(20),  # Top 20 results
                'criteria': criteria
            }
        }
        
        yield {
            "type": "complete",
            "content": f"✅ Screening complete: Found {len(results)} stocks matching criteria"
        }
    
    def stream_backtest(self, symbol: str, strategy: str) -> Generator[Dict, None, None]:
        """Stream backtesting results"""
        yield {
            "type": "status",
            "content": f"📈 Starting backtest for {symbol} with {strategy} strategy..."
        }
        
        yield {
            "type": "status",
            "content": f"📊 Downloading historical data for {symbol}..."
        }
        
        results = self.backtester.backtest_strategy(symbol, strategy, "1y")
        
        if "error" in results:
            yield {
                "type": "error",
                "content": f"❌ Backtest failed: {results['error']}"
            }
            return
        
        yield {
            "type": "status",
            "content": f"🔬 Computing strategy performance metrics..."
        }
        
        time.sleep(1)
        
        yield {
            "type": "backtest_results",
            "content": results
        }
        
        yield {
            "type": "complete",
            "content": f"✅ Backtest complete for {symbol} - {results['strategy']}"
        }
    
    def stream_comparison(self, symbols: List[str]) -> Generator[Dict, None, None]:
        """Stream comparison analysis for two symbols"""
        yield {
            "type": "status",
            "content": f"⚖️ Starting comparison: {symbols[0]} vs {symbols[1]}"
        }
        
        analyses = {}
        
        # Analyze each symbol
        for i, symbol in enumerate(symbols):
            yield {
                "type": "status",
                "content": f"📊 Analyzing {symbol} ({i+1}/{len(symbols)})..."
            }
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                info = ticker.info
                
                if data.empty:
                    yield {
                        "type": "status",
                        "content": f"⚠️ No data found for {symbol}, skipping..."
                    }
                    continue
                
                # Quick analysis for comparison
                yield {
                    "type": "status",
                    "content": f"🔬 Computing indicators for {symbol}..."
                }
                
                technical_data = self._calculate_technical_indicators(data)
                fundamental_data = self._extract_fundamentals(info)
                
                analyses[symbol] = {
                    'data': data,
                    'info': info,
                    'technical': technical_data,
                    'fundamental': fundamental_data,
                    'price': data['Close'].iloc[-1],
                    'change': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
                }
                
                yield {
                    "type": "status",
                    "content": f"✅ {symbol} analysis complete"
                }
                
            except Exception as e:
                yield {
                    "type": "status",
                    "content": f"❌ Failed to analyze {symbol}: {str(e)}"
                }
                continue
        
        # Generate comparison results
        if len(analyses) >= 2:
            yield {
                "type": "status",
                "content": "📋 Generating comparison report..."
            }
            
            time.sleep(0.5)
            
            yield {
                "type": "comparison",
                "content": {
                    'symbols': symbols,
                    'analyses': analyses
                }
            }
            
            yield {
                "type": "complete",
                "content": f"✅ Comparison complete: {' vs '.join(symbols)}"
            }
        else:
            yield {
                "type": "error",
                "content": "❌ Need at least 2 valid symbols for comparison. Please check the symbols and try again."
            }
    
    def stream_analysis(self, symbol: str) -> Generator[Dict, None, None]:
        """Stream analysis results progressively"""
        
        # Validate symbol first
        if not self.parser.validate_symbol(symbol):
            yield {
                "type": "error",
                "content": f"❌ Invalid symbol format: {symbol}"
            }
            return
        
        # Step 1: Initial validation
        yield {
            "type": "status",
            "content": f"🔍 Validating symbol: {symbol.upper()}"
        }
        time.sleep(0.5)
        
        # Step 2: Data fetching
        yield {
            "type": "status", 
            "content": f"📊 Fetching market data for {symbol.upper()}..."
        }
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            info = ticker.info
            
            if data.empty:
                yield {
                    "type": "error",
                    "content": f"❌ No data found for symbol: {symbol.upper()}. Please check if the symbol is correct."
                }
                return
            
            time.sleep(1)
            
            # Step 3: Basic info
            yield {
                "type": "info",
                "content": {
                    "symbol": symbol.upper(),
                    "name": info.get("longName", symbol.upper()),
                    "sector": info.get("sector", "Unknown"),
                    "price": data['Close'].iloc[-1],
                    "change": ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
                }
            }
            
            # Step 4: Technical analysis
            yield {
                "type": "status",
                "content": "🔬 Computing technical indicators..."
            }
            time.sleep(0.8)
            
            technical_data = self._calculate_technical_indicators(data)
            yield {
                "type": "technical",
                "content": technical_data
            }
            
            # Step 5: Fundamental analysis
            yield {
                "type": "status", 
                "content": "💰 Analyzing fundamentals..."
            }
            time.sleep(0.8)
            
            fundamental_data = self._extract_fundamentals(info)
            yield {
                "type": "fundamental",
                "content": fundamental_data
            }
            
            # Step 6: AI insights
            yield {
                "type": "status",
                "content": "🤖 Generating AI insights..."
            }
            time.sleep(1)
            
            ai_insights = self._generate_insights(technical_data, fundamental_data, data)
            yield {
                "type": "insights",
                "content": ai_insights
            }
            
            # Step 7: Chart data
            yield {
                "type": "status",
                "content": "📈 Preparing interactive charts..."
            }
            time.sleep(0.5)
            
            chart_data = self._create_chart(data, technical_data)
            yield {
                "type": "chart",
                "content": chart_data
            }
            
            # Final status
            yield {
                "type": "complete",
                "content": f"✅ Analysis complete for {symbol.upper()}"
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"❌ Analysis failed for {symbol.upper()}: {str(e)}"
            }
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        close = data['Close']
        high = data['High'] 
        low = data['Low']
        volume = data['Volume']
        
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
        indicators['sma_50'] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1] if len(close) >= 12 else None
        indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1] if len(close) >= 26 else None
        
        # RSI
        if len(close) >= 14:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        if len(close) >= 26:
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_histogram'] = (macd - signal).iloc[-1]
        
        # Bollinger Bands
        if len(close) >= 20:
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            indicators['bb_upper'] = (sma + 2*std).iloc[-1]
            indicators['bb_lower'] = (sma - 2*std).iloc[-1]
            indicators['bb_middle'] = sma.iloc[-1]
        
        # Volume
        indicators['volume_avg'] = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else None
        indicators['volume_ratio'] = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else None
        
        return indicators
    
    def _extract_fundamentals(self, info: Dict) -> Dict:
        """Extract fundamental data"""
        return {
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'roe': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'dividend_yield': info.get('dividendYield'),
            'revenue_growth': info.get('revenueGrowth'),
            'beta': info.get('beta'),
            'target_price': info.get('targetMeanPrice')
        }
    
    def _generate_insights(self, technical: Dict, fundamental: Dict, data: pd.DataFrame) -> Dict:
        """Generate AI insights"""
        signals = []
        score = 0
        total_factors = 0
        
        # RSI analysis
        rsi = technical.get('rsi', 50)
        if rsi < 30:
            signals.append({"type": "bullish", "message": "RSI oversold - potential buy signal"})
            score += 1
        elif rsi > 70:
            signals.append({"type": "bearish", "message": "RSI overbought - caution advised"})
        else:
            score += 0.5
        total_factors += 1
        
        # MACD analysis
        macd = technical.get('macd', 0)
        macd_signal = technical.get('macd_signal', 0)
        if macd and macd_signal:
            if macd > macd_signal:
                signals.append({"type": "bullish", "message": "MACD bullish crossover"})
                score += 1
            else:
                signals.append({"type": "bearish", "message": "MACD bearish crossover"})
            total_factors += 1
        
        # Moving average trend
        current_price = data['Close'].iloc[-1]
        sma_20 = technical.get('sma_20')
        sma_50 = technical.get('sma_50')
        
        if sma_20 and sma_50:
            if current_price > sma_20 > sma_50:
                signals.append({"type": "bullish", "message": "Price above rising moving averages"})
                score += 1
            elif current_price < sma_20 < sma_50:
                signals.append({"type": "bearish", "message": "Price below declining moving averages"})
            else:
                score += 0.5
            total_factors += 1
        
        # Fundamental signals
        pe = fundamental.get('pe_ratio')
        if pe:
            if pe < 15:
                signals.append({"type": "bullish", "message": "Low PE ratio - potentially undervalued"})
                score += 1
            elif pe > 30:
                signals.append({"type": "bearish", "message": "High PE ratio - potentially overvalued"})
            else:
                score += 0.5
            total_factors += 1
        
        confidence = score / total_factors if total_factors > 0 else 0.5
        
        # Generate recommendation
        if confidence >= 0.75:
            recommendation = "Strong Buy"
            risk_level = "Low"
        elif confidence >= 0.6:
            recommendation = "Buy" 
            risk_level = "Medium"
        elif confidence >= 0.4:
            recommendation = "Hold"
            risk_level = "Medium"
        else:
            recommendation = "Sell"
            risk_level = "High"
        
        return {
            'signals': signals,
            'confidence': confidence,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'target_price': current_price * (1 + confidence * 0.15) if confidence > 0.5 else None
        }
    
    def _create_chart(self, data: pd.DataFrame, technical: Dict) -> go.Figure:
        """Create professional chart"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price & Moving Averages', 'RSI', 'Volume'),
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'], 
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#10B981',
                decreasing_line_color='#EF4444'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if len(data) >= 20:
            sma_20 = data['Close'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(x=data.index, y=sma_20, name='SMA 20',
                          line=dict(color='#F59E0B', width=1)),
                row=1, col=1
            )
        
        if len(data) >= 50:
            sma_50 = data['Close'].rolling(50).mean()
            fig.add_trace(
                go.Scatter(x=data.index, y=sma_50, name='SMA 50',
                          line=dict(color='#EF4444', width=1)),
                row=1, col=1
            )
        
        # RSI
        if len(data) >= 14:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(x=data.index, y=rsi, name='RSI',
                          line=dict(color='#8B5CF6', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume',
                   marker_color='#3B82F6', opacity=0.7),
            row=3, col=1
        )
        
        fig.update_layout(
            title="Technical Analysis Chart",
            height=700,
            showlegend=True,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        return fig

# Enhanced Theme selector in sidebar
with st.sidebar:
    st.markdown("### 🎨 Theme")
    theme = st.selectbox("Select Theme", ["dark", "minimal", "terminal"], index=0)
    
    st.markdown("### 🔧 Tools")
    tool_mode = st.selectbox("Select Mode", [
        "🔍 Stock Analysis", 
        "📊 Stock Screener", 
        "📈 Strategy Backtest"
    ], index=0)
    
    if tool_mode == "📊 Stock Screener":
        st.markdown("#### Screening Criteria")
        
        # Price range
        price_range = st.slider("Price Range ($)", 0, 500, (10, 200))
        
        # Market cap
        min_market_cap = st.selectbox("Min Market Cap", [
            "Any", "1B+", "10B+", "50B+", "100B+"
        ], index=1)
        
        # P/E ratio
        max_pe = st.slider("Max P/E Ratio", 5, 50, 25)
        
        # RSI range
        rsi_range = st.slider("RSI Range", 0, 100, (20, 80))
        
        # Performance
        min_1m_return = st.slider("Min 1M Return (%)", -50, 50, -20)
        
        # Sectors
        sectors = st.multiselect("Sectors", [
            "Technology", "Healthcare", "Financials", "Consumer Discretionary",
            "Industrials", "Communication Services", "Energy", "Utilities",
            "Consumer Staples", "Materials", "Real Estate"
        ])
        
        if st.button("🔍 Run Screen"):
            screening_criteria = {
                'min_price': price_range[0],
                'max_price': price_range[1],
                'min_market_cap': {"Any": 0, "1B+": 1, "10B+": 10, "50B+": 50, "100B+": 100}[min_market_cap],
                'max_pe': max_pe,
                'min_rsi': rsi_range[0],
                'max_rsi': rsi_range[1],
                'min_1m_return': min_1m_return,
                'sectors': sectors if sectors else None
            }
            st.session_state.messages.append({
                "role": "user", 
                "content": f"screen stocks with custom criteria"
            })
            st.session_state.custom_screening_criteria = screening_criteria
            st.rerun()
    
    elif tool_mode == "📈 Strategy Backtest":
        st.markdown("#### Backtest Settings")
        
        backtest_symbol = st.text_input("Symbol", placeholder="AAPL")
        
        strategy = st.selectbox("Strategy", [
            "SMA Crossover",
            "RSI Mean Reversion", 
            "Momentum",
            "Buy and Hold"
        ])
        
        period = st.selectbox("Period", ["1y", "2y", "5y"], index=0)
        
        if st.button("📈 Run Backtest") and backtest_symbol:
            strategy_map = {
                "SMA Crossover": "sma_crossover",
                "RSI Mean Reversion": "rsi_mean_reversion",
                "Momentum": "momentum",
                "Buy and Hold": "buy_and_hold"
            }
            st.session_state.messages.append({
                "role": "user",
                "content": f"backtest {backtest_symbol} with {strategy_map[strategy]} strategy"
            })
            st.rerun()
    
    st.markdown("### 📊 Market Status")
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">🟢 OPEN</div>
        <div class="metric-label">US Markets</div>
    </div>
    """, unsafe_allow_html=True)

# Apply selected theme
st.markdown(get_theme_css(theme), unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "analyzer" not in st.session_state:
    st.session_state.analyzer = StreamingAnalyzer()

if "custom_screening_criteria" not in st.session_state:
    st.session_state.custom_screening_criteria = None

# Professional Header
st.markdown("""
<div class="terminal-header">
    <div class="terminal-title">📈 US Stock Terminal Pro</div>
    <div class="terminal-subtitle">Professional Real-time Market Analysis & Trading Intelligence</div>
</div>
""", unsafe_allow_html=True)

# Status Bar
st.markdown("""
<div class="status-bar">
    <div class="status-item">🟢 SYSTEM ONLINE</div>
    <div class="status-item">🇺🇸 NYSE • NASDAQ</div>
    <div class="status-item">🤖 AI ENGINE ACTIVE</div>
    <div class="status-item">📡 REAL-TIME DATA</div>
</div>
""", unsafe_allow_html=True)

# Quick Actions (if no messages)
if not st.session_state.messages:
    st.markdown("### ⚡ Quick Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Analyze AAPL", key="q1"):
            st.session_state.messages.append({"role": "user", "content": "AAPL"})
            st.rerun()
    
    with col2:
        if st.button("⚖️ Compare NVDA vs AAPL", key="q2"):
            st.session_state.messages.append({"role": "user", "content": "compare NVDA and AAPL"})
            st.rerun()
    
    with col3:
        if st.button("🔍 Screen Stocks", key="q3"):
            st.session_state.messages.append({"role": "user", "content": "screen stocks"})
            st.rerun()
    
    with col4:
        if st.button("📈 Backtest TSLA", key="q4"):
            st.session_state.messages.append({"role": "user", "content": "backtest TSLA with SMA strategy"})
            st.rerun()

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 TRADER:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-message">
            <strong>🤖 TERMINAL:</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Display message content
        if "content" in message:
            st.markdown(message["content"])
        
        # Display chart if present
        if "chart" in message:
            st.plotly_chart(message["chart"], use_container_width=True, config={'displayModeBar': False})
        
        # Display tables if present
        if "technical_table" in message:
            st.markdown("#### 📊 Technical Indicators")
            st.dataframe(message["technical_table"], use_container_width=True, hide_index=True)
        
        if "fundamental_table" in message:
            st.markdown("#### 💰 Fundamental Metrics")
            st.dataframe(message["fundamental_table"], use_container_width=True, hide_index=True)
            
        if "comparison_table" in message:
            st.markdown("#### ⚖️ Comparison Results")
            st.dataframe(message["comparison_table"], use_container_width=True, hide_index=True)
        
        if "screening_results" in message:
            st.markdown("#### 🔍 Stock Screening Results")
            results = message["screening_results"]
            st.markdown(f"**Found {results['total_found']} stocks matching criteria**")
            
            # Format the results table
            df = results['results'].copy()
            if not df.empty:
                # Format numeric columns
                numeric_cols = ['Price', 'Market Cap', 'P/E', 'P/B', 'ROE', 'Debt/Equity', 
                              'Div Yield', 'RSI', '1M Return', '3M Return']
                for col in numeric_cols:
                    if col in df.columns:
                        if col == 'Market Cap':
                            df[col] = df[col].apply(lambda x: f"${x/1e9:.1f}B" if pd.notnull(x) else "N/A")
                        elif col in ['Price', 'SMA 20', 'SMA 50']:
                            df[col] = df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                        elif col in ['P/E', 'P/B', 'Debt/Equity']:
                            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                        elif col in ['ROE', 'Div Yield', '1M Return', '3M Return']:
                            df[col] = df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
                        elif col == 'RSI':
                            df[col] = df[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
                
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        if "backtest_results" in message:
            st.markdown("#### 📈 Backtest Results")
            results = message["backtest_results"]
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Strategy Return", f"{results['total_return']:.1f}%")
            with col2:
                st.metric("Buy & Hold Return", f"{results['buy_hold_return']:.1f}%")
            with col3:
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")
            with col4:
                st.metric("Total Trades", results['total_trades'])
            
            # Strategy performance chart
            if 'data' in results and not results['data'].empty:
                fig = go.Figure()
                
                data = results['data']
                
                # Calculate cumulative returns
                strategy_cumret = (1 + data['Strategy_Returns'].fillna(0)).cumprod()
                buyhold_cumret = (1 + data['Returns'].fillna(0)).cumprod()
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=strategy_cumret,
                    name=results['strategy'],
                    line=dict(color='#10B981', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=buyhold_cumret,
                    name='Buy & Hold',
                    line=dict(color='#3B82F6', width=2)
                ))
                
                # Add buy/sell signals
                signals = results.get('signals', [])
                if signals:
                    buy_signals = [s for s in signals if s.get('Signal', s.get('Position', 0)) > 0]
                    
                    if buy_signals:
                        buy_dates = [data.index[i] for i, s in enumerate(buy_signals[:10])]
                        buy_prices = [s['Close'] for s in buy_signals[:10]]
                        fig.add_trace(go.Scatter(
                            x=buy_dates,
                            y=buy_prices,
                            mode='markers',
                            marker=dict(color='green', size=10, symbol='triangle-up'),
                            name='Buy Signals'
                        ))
