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
from typing import Dict, List, Optional, Generator, Any, Tuple
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

class StockDatabase:
    """Enhanced stock database with comprehensive US stock listings"""
    
    def __init__(self):
        self.stocks_cache = None
        self.last_update = None
        self.cache_duration = 3600  # 1 hour
        
    def get_all_us_stocks(self) -> pd.DataFrame:
        """Get comprehensive list of US stocks"""
        current_time = time.time()
        
        # Use cache if available and not expired
        if (self.stocks_cache is not None and 
            self.last_update is not None and 
            current_time - self.last_update < self.cache_duration):
            return self.stocks_cache
        
        try:
            # Get stocks from multiple exchanges - Using a curated list for reliability
            stock_symbols = [
                # Tech Giants
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE',
                # Large Cap
                'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'CRM',
                'KO', 'PFE', 'INTC', 'VZ', 'WMT', 'T', 'XOM', 'CVX', 'MRK', 'ABT', 'COST', 'PEP',
                'TMO', 'ACN', 'AVGO', 'TXN', 'LLY', 'NKE', 'MDT', 'DHR', 'UPS', 'QCOM', 'NEE',
                'PM', 'IBM', 'LIN', 'LOW', 'HON', 'AMGN', 'SBUX', 'UNP', 'C', 'GS', 'SPGI', 'BLK',
                'CAT', 'ISRG', 'AXP', 'BKNG', 'GILD', 'MMM', 'SYK', 'TJX', 'CVS', 'MO', 'USB',
                'ADP', 'MDLZ', 'TMUS', 'CI', 'SO', 'ZTS', 'DUK', 'CL', 'GE', 'D', 'ICE',
                # Mid Cap & Growth
                'AMD', 'MRNA', 'ZM', 'DOCU', 'ROKU', 'PTON', 'SQ', 'SHOP', 'TWLO', 'OKTA', 'SNOW',
                'PLTR', 'RBLX', 'HOOD', 'COIN', 'RIVN', 'LCID', 'F', 'GM', 'UBER', 'LYFT', 'ABNB',
                'DASH', 'DKNG', 'PENN', 'FUBO', 'CRSR', 'CRWD', 'ZS', 'NET', 'DDOG', 'ESTC', 'MDB',
                'TEAM', 'WDAY', 'SPLK', 'NOW', 'PANW', 'FTNT', 'CYBR',
                # ETFs
                'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'ARKK', 'XLF', 'XLE', 'XLK', 'XLV',
                # Additional stocks starting with common prefixes
                'TECH', 'TECHNO', 'TESLA', 'MICRO', 'MICROSOFT', 'APPLE', 'AMAZON', 'TESLA'
            ]
            
            # Create DataFrame with exchange info
            all_stocks = []
            for symbol in stock_symbols:
                if symbol.startswith(('SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'ARKK', 'XL')):
                    exchange = 'ETF'
                elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX']:
                    exchange = 'NASDAQ'
                else:
                    exchange = 'NYSE'
                
                all_stocks.append({'Symbol': symbol, 'Exchange': exchange})
            
            stocks_df = pd.DataFrame(all_stocks)
            self.stocks_cache = stocks_df
            self.last_update = current_time
            return stocks_df
            
        except Exception as e:
            st.error(f"Error loading stock database: {e}")
            # Return minimal fallback
            fallback_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
            return pd.DataFrame([{'Symbol': s, 'Exchange': 'Mixed'} for s in fallback_stocks])

class EnhancedCommandParser:
    """Enhanced command parser for stock analysis, screening, and comparison"""
    
    @staticmethod
    def parse_command(text: str) -> Dict:
        """Parse user command and extract symbols, action, and parameters"""
        text = text.upper().strip()
        
        # Stock symbol pattern (1-5 letters, optionally followed by numbers or dash)
        symbol_pattern = r'\b[A-Z]{1,5}(?:[-\d]{0,3})?\b'
        
        # Extract all potential symbols
        potential_symbols = re.findall(symbol_pattern, text)
        
        # Filter out common command words
        command_words = {
            'ANALYZE', 'ANALYSE', 'CHECK', 'LOOK', 'AT', 'SHOW', 'ME', 'TELL', 'ABOUT', 'GET', 
            'DATA', 'FOR', 'COMPARE', 'VS', 'VERSUS', 'AND', 'WITH', 'AGAINST', 'STOCK', 'STOCKS', 
            'PRICE', 'CHART', 'INFO', 'INFORMATION', 'DETAILS', 'ANALYSIS', 'REPORT', 'THE', 'OF', 
            'IS', 'ARE', 'WHAT', 'HOW', 'WHY', 'WHEN', 'WHERE', 'WHICH', 'SCREEN', 'SCREENING',
            'FILTER', 'FIND', 'SEARCH', 'SCAN', 'ALL', 'TOP', 'BEST', 'WORST', 'HIGH', 'LOW',
            'ABOVE', 'BELOW', 'GREATER', 'LESS', 'THAN', 'EQUAL', 'TO', 'RATIO', 'YIELD',
            'GROWTH', 'VALUE', 'MOMENTUM', 'VOLUME', 'MARKET', 'CAP', 'DIVIDEND'
        }
        
        symbols = [s for s in potential_symbols if s not in command_words]
        
        # Determine action type
        action = 'analyze'  # Default
        parameters = {}
        
        if any(word in text for word in ['SCREEN', 'SCREENING', 'FILTER', 'FIND', 'SEARCH', 'SCAN']):
            action = 'screen'
            
            # Parse screening criteria
            if 'PE' in text or 'P/E' in text:
                if 'BELOW' in text or 'LESS' in text or '<' in text:
                    pe_match = re.search(r'(?:PE|P/E).*?(?:BELOW|LESS|<)\s*(\d+(?:\.\d+)?)', text)
                    if pe_match:
                        parameters['pe_max'] = float(pe_match.group(1))
                elif 'ABOVE' in text or 'GREATER' in text or '>' in text:
                    pe_match = re.search(r'(?:PE|P/E).*?(?:ABOVE|GREATER|>)\s*(\d+(?:\.\d+)?)', text)
                    if pe_match:
                        parameters['pe_min'] = float(pe_match.group(1))
            
            if 'MARKET CAP' in text or 'MARKETCAP' in text:
                if 'ABOVE' in text:
                    cap_match = re.search(r'MARKET\s*CAP.*?ABOVE\s*(\d+(?:\.\d+)?)[BM]?', text)
                    if cap_match:
                        multiplier = 1e9 if 'B' in text else 1e6 if 'M' in text else 1e9
                        parameters['market_cap_min'] = float(cap_match.group(1)) * multiplier
            
            if 'RSI' in text:
                if 'BELOW' in text or 'OVERSOLD' in text:
                    parameters['rsi_max'] = 30
                elif 'ABOVE' in text or 'OVERBOUGHT' in text:
                    parameters['rsi_min'] = 70
            
            if 'DIVIDEND' in text and 'YIELD' in text:
                if 'ABOVE' in text:
                    div_match = re.search(r'DIVIDEND.*?YIELD.*?ABOVE\s*(\d+(?:\.\d+)?)', text)
                    if div_match:
                        parameters['dividend_yield_min'] = float(div_match.group(1)) / 100
            
            if 'VOLUME' in text and 'HIGH' in text:
                parameters['high_volume'] = True
                
        elif any(word in text for word in ['COMPARE', 'VS', 'VERSUS', 'AGAINST']):
            action = 'compare'
            
        elif '*' in text:
            action = 'check_all'
            # Extract the symbol before *
            star_match = re.search(r'([A-Z]+)\*', text)
            if star_match:
                parameters['prefix'] = star_match.group(1)
        
        return {
            'action': action,
            'symbols': symbols[:10],  # Limit to 10 symbols max
            'parameters': parameters,
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
        
        # Rest can be letters, numbers, or dash
        return all(c.isalnum() or c == '-' for c in symbol)

class StreamingAnalyzer:
    """Enhanced Streaming Analysis Engine"""
    
    def __init__(self):
        self.data_cache = {}
        self.parser = EnhancedCommandParser()
        self.stock_db = StockDatabase()
        
    def process_command(self, text: str) -> Generator[Dict, None, None]:
        """Process user command and route to appropriate analysis"""
        # Parse the command
        parsed = self.parser.parse_command(text)
        
        yield {
            "type": "status",
            "content": f"🔍 Processing command: {parsed['original_text']}"
        }
        
        # Route to appropriate analysis
        if parsed['action'] == 'screen':
            yield from self.stream_screening(parsed['parameters'])
        elif parsed['action'] == 'compare' and len(parsed['symbols']) >= 2:
            yield from self.stream_comparison(parsed['symbols'])
        elif parsed['action'] == 'check_all':
            yield from self.stream_check_all(parsed['parameters'])
        elif parsed['symbols']:
            # Single or multiple symbol analysis
            if len(parsed['symbols']) == 1:
                yield from self.stream_single_analysis(parsed['symbols'][0])
            else:
                yield from self.stream_multiple_analysis(parsed['symbols'])
        else:
            yield {
                "type": "error",
                "content": "❌ No valid command or symbols found. Try: AAPL, screen PE < 15, compare AAPL MSFT, TECH*"
            }
    
    def stream_single_analysis(self, symbol: str) -> Generator[Dict, None, None]:
        """Stream analysis for a single stock symbol"""
        yield {
            "type": "status",
            "content": f"🔍 Validating symbol: {symbol}"
        }
        time.sleep(0.5)

        # Step 2: Fetch data
        yield {
            "type": "status",
            "content": f"📊 Fetching data for {symbol}..."
        }
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            info = ticker.info

            if data.empty:
                yield {
                    "type": "error",
                    "content": f"⚠️ No data found for {symbol}"
                }
                return

            # Step 3: Technical analysis
            yield {
                "type": "status",
                "content": f"🔬 Calculating technical indicators for {symbol}..."
            }
            technical_data = self._calculate_technical_indicators(data)

            # Step 4: Fundamental analysis
            yield {
                "type": "status",
                "content": f"💰 Analyzing fundamentals for {symbol}..."
            }
            fundamental_data = self._extract_fundamentals(info)

            # Step 5: Performance metrics
            yield {
                "type": "status",
                "content": f"📈 Calculating performance metrics for {symbol}..."
            }
            performance_data = self._calculate_performance_metrics(data)

            # Step 6: Generate insights
            yield {
                "type": "status",
                "content": f"🤖 Generating AI insights for {symbol}..."
            }
            insights = self._generate_insights(technical_data, fundamental_data, data)

            # Step 7: Create chart
            yield {
                "type": "status",
                "content": f"📊 Preparing chart for {symbol}..."
            }
            chart = self._create_chart(data, technical_data)

            # Step 8: Return results
            yield {
                "type": "analysis",
                "content": {
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'price': data['Close'].iloc[-1],
                    'change': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0,
                    'technical': technical_data,
                    'fundamental': fundamental_data,
                    'performance': performance_data,
                    'insights': insights,
                    'chart': chart
                }
            }

            yield {
                "type": "complete",
                "content": f"✅ Analysis complete for {symbol}"
            }

        except Exception as e:
            yield {
                "type": "error",
                "content": f"❌ Failed to analyze {symbol}: {str(e)}"
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
        indicators['sma_200'] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
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
            indicators['bb_width'] = ((indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle'] * 100)
        
        # Stochastic Oscillator
        if len(close) >= 14:
            low_14 = low.rolling(14).min()
            high_14 = high.rolling(14).max()
            k_percent = 100 * ((close - low_14) / (high_14 - low_14))
            indicators['stoch_k'] = k_percent.iloc[-1]
            indicators['stoch_d'] = k_percent.rolling(3).mean().iloc[-1]
        
        # Volume indicators
        indicators['volume_avg'] = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else None
        indicators['volume_ratio'] = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else None
        
        # Support and Resistance levels
        if len(close) >= 20:
            recent_highs = high.rolling(20).max()
            recent_lows = low.rolling(20).min()
            indicators['resistance'] = recent_highs.iloc[-1]
            indicators['support'] = recent_lows.iloc[-1]
        
        return indicators
    
    def _extract_fundamentals(self, info: Dict) -> Dict:
        """Extract fundamental data"""
        return {
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'ev_revenue': info.get('enterpriseToRevenue'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'beta': info.get('beta'),
            'target_price': info.get('targetMeanPrice'),
            'recommendation': info.get('recommendationMean'),
            'analyst_count': info.get('numberOfAnalystOpinions'),
            'insider_ownership': info.get('heldPercentInsiders'),
            'institutional_ownership': info.get('heldPercentInstitutions'),
            'short_ratio': info.get('shortRatio'),
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'gross_margin': info.get('grossMargins')
        }
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        close = data['Close']
        
        metrics = {}
        
        # Returns
        if len(close) > 1:
            metrics['1d_return'] = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
        
        if len(close) >= 5:
            metrics['1w_return'] = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100)
        
        if len(close) >= 22:
            metrics['1m_return'] = ((close.iloc[-1] - close.iloc[-22]) / close.iloc[-22] * 100)
        
        if len(close) >= 66:
            metrics['3m_return'] = ((close.iloc[-1] - close.iloc[-66]) / close.iloc[-66] * 100)
        
        if len(close) >= 132:
            metrics['6m_return'] = ((close.iloc[-1] - close.iloc[-132]) / close.iloc[-132] * 100)
        
        metrics['ytd_return'] = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100)
        
        # Volatility (standard deviation of daily returns)
        if len(close) > 1:
            daily_returns = close.pct_change().dropna()
            metrics['volatility'] = daily_returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            excess_returns = daily_returns - 0.02/252
            if excess_returns.std() != 0:
                metrics['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Max drawdown
        if len(close) > 1:
            cumulative = (1 + close.pct_change()).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            metrics['max_drawdown'] = drawdown.min() * 100
        
        return metrics
    
    def _generate_insights(self, technical: Dict, fundamental: Dict, data: pd.DataFrame) -> Dict:
        """Generate enhanced AI insights"""
        signals = []
        score = 0
        total_factors = 0
        
        current_price = data['Close'].iloc[-1]
        
        # Technical Analysis Signals
        
        # RSI analysis
        rsi = technical.get('rsi', 50)
        if rsi < 30:
            signals.append({"type": "bullish", "strength": "strong", "message": "RSI oversold - strong buy signal"})
            score += 2
        elif rsi < 40:
            signals.append({"type": "bullish", "strength": "moderate", "message": "RSI approaching oversold - potential buy"})
            score += 1
        elif rsi > 70:
            signals.append({"type": "bearish", "strength": "strong", "message": "RSI overbought - caution advised"})
        elif rsi > 60:
            signals.append({"type": "neutral", "strength": "weak", "message": "RSI approaching overbought territory"})
            score += 0.5
        else:
            score += 1
        total_factors += 2
        
        # MACD analysis
        macd = technical.get('macd', 0)
        macd_signal = technical.get('macd_signal', 0)
        macd_hist = technical.get('macd_histogram', 0)
        
        if macd and macd_signal:
            if macd > macd_signal and macd_hist > 0:
                signals.append({"type": "bullish", "strength": "moderate", "message": "MACD bullish crossover with positive momentum"})
                score += 1.5
            elif macd > macd_signal:
                signals.append({"type": "bullish", "strength": "weak", "message": "MACD above signal line"})
                score += 1
            else:
                signals.append({"type": "bearish", "strength": "moderate", "message": "MACD bearish crossover"})
            total_factors += 1.5
        
        # Moving average trend analysis
        sma_20 = technical.get('sma_20')
        sma_50 = technical.get('sma_50')
        sma_200 = technical.get('sma_200')
        
        if sma_20 and sma_50 and sma_200:
            if current_price > sma_20 > sma_50 > sma_200:
                signals.append({"type": "bullish", "strength": "strong", "message": "Strong uptrend - all moving averages aligned bullishly"})
                score += 2
            elif current_price > sma_20 > sma_50:
                signals.append({"type": "bullish", "strength": "moderate", "message": "Price above short and medium-term moving averages"})
                score += 1.5
            elif current_price < sma_20 < sma_50 < sma_200:
                signals.append({"type": "bearish", "strength": "strong", "message": "Strong downtrend - all moving averages aligned bearishly"})
            else:
                score += 0.5
            total_factors += 2
        
        # Bollinger Bands analysis
        bb_upper = technical.get('bb_upper')
        bb_lower = technical.get('bb_lower')
        bb_width = technical.get('bb_width')
        
        if bb_upper and bb_lower:
            if current_price <= bb_lower:
                signals.append({"type": "bullish", "strength": "moderate", "message": "Price at lower Bollinger Band - potential bounce"})
                score += 1
            elif current_price >= bb_upper:
                signals.append({"type": "bearish", "strength": "moderate", "message": "Price at upper Bollinger Band - potential pullback"})
            
            if bb_width and bb_width < 10:
                signals.append({"type": "neutral", "strength": "weak", "message": "Low volatility - potential breakout incoming"})
            
            total_factors += 1
        
        # Volume analysis
        volume_ratio = technical.get('volume_ratio', 1)
        if volume_ratio > 2:
            signals.append({"type": "bullish", "strength": "moderate", "message": "High volume confirms price movement"})
            score += 0.5
        elif volume_ratio < 0.5:
            signals.append({"type": "bearish", "strength": "weak", "message": "Low volume - weak conviction"})
        total_factors += 0.5
        
        # Fundamental Analysis Signals
        
        # Valuation metrics
        pe = fundamental.get('pe_ratio')
        if pe:
            if pe < 10:
                signals.append({"type": "bullish", "strength": "strong", "message": "Very low PE ratio - potentially undervalued"})
                score += 2
            elif pe < 15:
                signals.append({"type": "bullish", "strength": "moderate", "message": "Low PE ratio - potentially undervalued"})
                score += 1
            elif pe > 30:
                signals.append({"type": "bearish", "strength": "moderate", "message": "High PE ratio - potentially overvalued"})
            elif pe > 50:
                signals.append({"type": "bearish", "strength": "strong", "message": "Very high PE ratio - significantly overvalued"})
            else:
                score += 0.5
            total_factors += 1.5
        
        # Growth metrics
        revenue_growth = fundamental.get('revenue_growth')
        if revenue_growth:
            if revenue_growth > 0.20:
                signals.append({"type": "bullish", "strength": "strong", "message": "Strong revenue growth > 20%"})
                score += 1.5
            elif revenue_growth > 0.10:
                signals.append({"type": "bullish", "strength": "moderate", "message": "Good revenue growth > 10%"})
                score += 1
            elif revenue_growth < -0.10:
                signals.append({"type": "bearish", "strength": "moderate", "message": "Declining revenue - negative growth"})
            total_factors += 1
        
        # Financial health
        debt_to_equity = fundamental.get('debt_to_equity')
        current_ratio = fundamental.get('current_ratio')
        
        if debt_to_equity and debt_to_equity < 0.3:
            signals.append({"type": "bullish", "strength": "weak", "message": "Low debt levels - strong balance sheet"})
            score += 0.5
        elif debt_to_equity and debt_to_equity > 2:
            signals.append({"type": "bearish", "strength": "moderate", "message": "High debt levels - financial risk"})
        
        if current_ratio and current_ratio > 2:
            signals.append({"type": "bullish", "strength": "weak", "message": "Strong liquidity position"})
            score += 0.5
        elif current_ratio and current_ratio < 1:
            signals.append({"type": "bearish", "strength": "moderate", "message": "Poor liquidity - potential cash flow issues"})
        
        total_factors += 1
        
        # Profitability
        roe = fundamental.get('roe')
        if roe and roe > 0.15:
            signals.append({"type": "bullish", "strength": "moderate", "message": "High ROE > 15% - efficient management"})
            score += 1
        elif roe and roe < 0:
            signals.append({"type": "bearish", "strength": "strong", "message": "Negative ROE - unprofitable"})
        
        total_factors += 1
        
        # Calculate final metrics
        confidence = score / total_factors if total_factors > 0 else 0.5
        
        # Generate recommendation
        if confidence >= 0.8:
            recommendation = "Strong Buy"
            risk_level = "Low"
        elif confidence >= 0.65:
            recommendation = "Buy" 
            risk_level = "Low-Medium"
        elif confidence >= 0.5:
            recommendation = "Hold"
            risk_level = "Medium"
        elif confidence >= 0.35:
            recommendation = "Weak Hold"
            risk_level = "Medium-High"
        else:
            recommendation = "Sell"
            risk_level = "High"
        
        # Calculate target price
        target_multiplier = 1 + (confidence - 0.5) * 0.3  # -15% to +15% based on confidence
        target_price = current_price * target_multiplier if confidence != 0.5 else None
        
        return {
            'signals': signals,
            'confidence': confidence,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'target_price': target_price,
            'score': score,
            'total_factors': total_factors,
            'technical_score': sum(1 for s in signals if 'RSI' in s['message'] or 'MACD' in s['message'] or 'moving averages' in s['message']),
            'fundamental_score': sum(1 for s in signals if any(word in s['message'] for word in ['PE', 'revenue', 'debt', 'ROE']))
        }
    
    def _create_chart(self, data: pd.DataFrame, technical: Dict) -> go.Figure:
        """Create enhanced professional chart"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Price Action & Moving Averages', 'RSI & Stochastic', 'MACD', 'Volume'),
            vertical_spacing=0.06,
            row_heights=[0.5, 0.2, 0.15, 0.15]
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
        
        if len(data) >= 200:
            sma_200 = data['Close'].rolling(200).mean()
            fig.add_trace(
                go.Scatter(x=data.index, y=sma_200, name='SMA 200',
                          line=dict(color='#8B5CF6', width=2)),
                row=1, col=1
            )
        
        # Bollinger Bands
        if len(data) >= 20:
            sma = data['Close'].rolling(20).mean()
            std = data['Close'].rolling(20).std()
            bb_upper = sma + 2*std
            bb_lower = sma - 2*std
            
            fig.add_trace(
                go.Scatter(x=data.index, y=bb_upper, name='BB Upper',
                          line=dict(color='#6B7280', width=1, dash='dash'),
                          fill=None),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=bb_lower, name='BB Lower',
                          line=dict(color='#6B7280', width=1, dash='dash'),
                          fill='tonexty', fillcolor='rgba(107, 114, 128, 0.1)'),
                row=1, col=1
            )
        
        # RSI and Stochastic
        if len(data) >= 14:
            # RSI
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
            
            # Stochastic
            low_14 = data['Low'].rolling(14).min()
            high_14 = data['High'].rolling(14).max()
            k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(3).mean()
            
            fig.add_trace(
                go.Scatter(x=data.index, y=k_percent, name='Stoch %K',
                          line=dict(color='#F59E0B', width=1)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=d_percent, name='Stoch %D',
                          line=dict(color='#10B981', width=1)),
                row=2, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD
        if len(data) >= 26:
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            fig.add_trace(
                go.Scatter(x=data.index, y=macd, name='MACD',
                          line=dict(color='#3B82F6', width=2)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=signal, name='Signal',
                          line=dict(color='#EF4444', width=1)),
                row=3, col=1
            )
            fig.add_trace(
                go.Bar(x=data.index, y=histogram, name='Histogram',
                       marker_color=['green' if x > 0 else 'red' for x in histogram],
                       opacity=0.7),
                row=3, col=1
            )
        
        # Volume
        colors = ['green' if close > open else 'red' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume',
                   marker_color=colors, opacity=0.7),
            row=4, col=1
        )
        
        # Volume moving average
        if len(data) >= 20:
            vol_ma = data['Volume'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(x=data.index, y=vol_ma, name='Vol MA',
                          line=dict(color='#F59E0B', width=1)),
                row=4, col=1
            )
        
        fig.update_layout(
            title="Professional Technical Analysis Chart",
            height=900,
            showlegend=True,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif"),
            xaxis_rangeslider_visible=False
        )
        
        return fig

    def stream_screening(self, criteria: Dict) -> Generator[Dict, None, None]:
        """Stream stock screening results"""
        yield {
            "type": "status",
            "content": "🔍 Screening stocks based on your criteria..."
        }
        
        # Get stock universe
        stocks_df = self.stock_db.get_all_us_stocks()
        
        yield {
            "type": "status",
            "content": f"📊 Analyzing {len(stocks_df)} stocks..."
        }
        
        results = []
        processed = 0
        
        # Process stocks in batches
        symbols = stocks_df['Symbol'].tolist()[:50]  # Limit for demo
        
        for symbol in symbols:
            processed += 1
            
            if processed % 10 == 0:
                yield {
                    "type": "status",
                    "content": f"🔄 Processed {processed}/{len(symbols)} stocks..."
                }
            
            try:
                result = self._analyze_for_screening(symbol, criteria)
                if result and self._meets_criteria(result, criteria):
                    results.append(result)
            except Exception as e:
                continue
        
        # Sort and format results
        if results:
            # Sort by score or PE ratio
            if results and 'pe_ratio' in results[0] and results[0]['pe_ratio']:
                results.sort(key=lambda x: x.get('pe_ratio', 999))
            
            yield {
                "type": "screening_results",
                "content": {
                    'results': results[:20],  # Top 20 results
                    'criteria': criteria,
                    'total_found': len(results)
                }
            }
        else:
            yield {
                "type": "error",
                "content": "❌ No stocks found matching your criteria. Try adjusting the parameters."
            }
    
    def stream_check_all(self, parameters: Dict) -> Generator[Dict, None, None]:
        """Stream analysis for all stocks matching a pattern"""
        prefix = parameters.get('prefix', '')
        
        yield {
            "type": "status",
            "content": f"🔍 Finding all stocks starting with '{prefix}'..."
        }
        
        # Get stock universe
        stocks_df = self.stock_db.get_all_us_stocks()
        
        # Filter by prefix
        if prefix:
            matching_stocks = stocks_df[stocks_df['Symbol'].str.startswith(prefix)]['Symbol'].tolist()
        else:
            matching_stocks = stocks_df['Symbol'].tolist()[:20]  # Limit for demo
        
        if not matching_stocks:
            yield {
                "type": "error",
                "content": f"❌ No stocks found starting with '{prefix}'"
            }
            return
        
        yield {
            "type": "status",
            "content": f"📊 Found {len(matching_stocks)} matching stocks. Analyzing..."
        }
        
        results = []
        for i, symbol in enumerate(matching_stocks[:10]):  # Limit to 10 for demo
            yield {
                "type": "status",
                "content": f"🔄 Analyzing {symbol} ({i+1}/{min(len(matching_stocks), 10)})..."
            }
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")
                info = ticker.info
                
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
                    
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol)[:40],
                        'price': current_price,
                        'change': change,
                        'volume': data['Volume'].iloc[-1],
                        'market_cap': info.get('marketCap'),
                        'pe_ratio': info.get('trailingPE'),
                        'sector': info.get('sector', 'Unknown')
                    })
                    
            except Exception as e:
                continue
        
        if results:
            yield {
                "type": "check_all_results",
                "content": {
                    'results': results,
                    'prefix': prefix,
                    'total_found': len(matching_stocks)
                }
            }
        else:
            yield {
                "type": "error",
                "content": f"❌ Could not analyze stocks starting with '{prefix}'"
            }
    
    def stream_multiple_analysis(self, symbols: List[str]) -> Generator[Dict, None, None]:
        """Stream analysis for multiple symbols"""
        yield {
            "type": "status",
            "content": f"📊 Analyzing {len(symbols)} stocks: {', '.join(symbols)}"
        }
        
        results = []
        for i, symbol in enumerate(symbols):
            yield {
                "type": "status",
                "content": f"🔄 Analyzing {symbol} ({i+1}/{len(symbols)})..."
            }
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                info = ticker.info
                
                if not data.empty:
                    technical_data = self._calculate_technical_indicators(data)
                    fundamental_data = self._extract_fundamentals(info)
                    
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol)[:40],
                        'price': data['Close'].iloc[-1],
                        'change': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0,
                        'technical': technical_data,
                        'fundamental': fundamental_data,
                        'sector': info.get('sector', 'Unknown')
                    })
                    
            except Exception as e:
                yield {
                    "type": "status",
                    "content": f"⚠️ Could not analyze {symbol}: {str(e)}"
                }
                continue
        
        if results:
            yield {
                "type": "multiple_analysis",
                "content": {
                    'results': results,
                    'symbols': symbols
                }
            }
        else:
            yield {
                "type": "error",
                "content": "❌ Could not analyze any of the provided symbols"
            }
    
    def _analyze_for_screening(self, symbol: str, criteria: Dict) -> Optional[Dict]:
        """Analyze a single stock for screening"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            info = ticker.info
            
            if data.empty:
                return None
            
            # Calculate key metrics
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1] if len(data) > 0 else 0
            
            # Calculate RSI if enough data
            rsi = None
            if len(data) >= 14:
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol)[:40],
                'price': current_price,
                'change': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0,
                'volume': volume,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'rsi': rsi,
                'sector': info.get('sector', 'Unknown'),
                'beta': info.get('beta')
            }
            
        except Exception as e:
            return None
    
    def _meets_criteria(self, stock_data: Dict, criteria: Dict) -> bool:
        """Check if stock meets screening criteria"""
        # PE ratio criteria
        if 'pe_max' in criteria and stock_data.get('pe_ratio'):
            if stock_data['pe_ratio'] > criteria['pe_max']:
                return False
        
        if 'pe_min' in criteria and stock_data.get('pe_ratio'):
            if stock_data['pe_ratio'] < criteria['pe_min']:
                return False
        
        # Market cap criteria
        if 'market_cap_min' in criteria and stock_data.get('market_cap'):
            if stock_data['market_cap'] < criteria['market_cap_min']:
                return False
        
        # RSI criteria
        if 'rsi_max' in criteria and stock_data.get('rsi'):
            if stock_data['rsi'] > criteria['rsi_max']:
                return False
        
        if 'rsi_min' in criteria and stock_data.get('rsi'):
            if stock_data['rsi'] < criteria['rsi_min']:
                return False

        # Dividend yield criteria
        if 'dividend_yield_min' in criteria and stock_data.get('dividend_yield'):
            if stock_data['dividend_yield'] < criteria['dividend_yield_min']:
                return False
        
        # High volume criteria
        if criteria.get('high_volume') and stock_data.get('volume'):
            # Consider high volume as > 1M shares
            if stock_data['volume'] < 1000000:
                return False
        
        return True

# Theme selector in sidebar
with st.sidebar:
    st.markdown("### 🎨 Theme")
    theme = st.selectbox("Select Theme", ["dark", "minimal", "terminal"], index=0)
    
    st.markdown("### 📊 Market Status")
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">🟢 OPEN</div>
        <div class="metric-label">US Markets</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Command Guide")
    st.markdown("""
    **Analysis Comm
