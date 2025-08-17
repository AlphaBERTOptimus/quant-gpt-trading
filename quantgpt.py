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
    
    /* Professional Footer */
    .professional-footer {{
        background: var(--bg-secondary);
        border: 1px solid var(--bg-accent);
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
    }}
    
    .footer-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }}
    
    .footer-subtitle {{
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        color: var(--text-secondary);
    }}
    
    .footer-stats {{
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        margin: 1.5rem 0;
    }}
    
    .footer-stat {{
        font-size: 0.85rem;
        color: var(--accent-color);
    }}
    
    .footer-commands {{
        margin: 1rem 0;
        font-size: 0.8rem;
        line-height: 1.5;
    }}
    
    .footer-disclaimer {{
        font-size: 0.75rem;
        margin-top: 1.5rem;
        opacity: 0.7;
    }}
    
    /* Code styling */
    code {{
        background: var(--bg-accent);
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85em;
        color: var(--accent-color);
    }}
    
    /* Responsive */
    @media (max-width: 768px) {{
        .terminal-header {{ padding: 1rem; }}
        .status-bar {{ flex-direction: column; gap: 1rem; }}
        .quick-action {{ margin: 0.25rem; }}
        .footer-stats {{ flex-direction: column; gap: 1rem; }}
    }}
</style>
"""

class CommandParser:
    """Smart command parser for stock analysis"""
    
    @staticmethod
    def parse_command(text: str) -> Dict:
        """Parse user command and extract symbols and action"""
        text = text.upper().strip()
        
        # Check for screening commands first
        if any(word in text for word in ['SCREEN', 'FIND', 'SEARCH', 'FILTER', 'LIST']):
            return CommandParser._parse_screen_command(text)
        
        # Stock symbol pattern (1-5 letters, optionally followed by numbers)
        symbol_pattern = r'\b[A-Z]{1,5}(?:\d{0,2})?\b'
        
        # Extract all potential symbols
        potential_symbols = re.findall(symbol_pattern, text)
        
        # Filter out common command words
        command_words = {'ANALYZE', 'ANALYSE', 'CHECK', 'LOOK', 'AT', 'SHOW', 'ME', 'TELL', 
                        'ABOUT', 'GET', 'DATA', 'FOR', 'COMPARE', 'VS', 'VERSUS', 'AND', 
                        'WITH', 'AGAINST', 'STOCK', 'STOCKS', 'PRICE', 'CHART', 'INFO',
                        'INFORMATION', 'DETAILS', 'ANALYSIS', 'REPORT', 'THE', 'OF', 'IS',
                        'ARE', 'WHAT', 'HOW', 'WHY', 'WHEN', 'WHERE', 'WHICH', 'SCREEN',
                        'FIND', 'SEARCH', 'FILTER', 'LIST', 'HIGH', 'LOW', 'DIVIDEND',
                        'GROWTH', 'VALUE', 'BLUE', 'CHIP'}
        
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
    def _parse_screen_command(text: str) -> Dict:
        """Parse screening commands"""
        text = text.upper().strip()
        
        # Screen for specific criteria (PE < 20, ROE > 15%, etc.)
        criteria_patterns = [
            r'(PE|P/E)\s*([<>=])\s*(\d+(?:\.\d+)?)',
            r'(PB|P/B)\s*([<>=])\s*(\d+(?:\.\d+)?)',
            r'(ROE)\s*([<>=])\s*(\d+(?:\.\d+)?)',
            r'(DEBT|DEBT/EQUITY|DEBT-TO-EQUITY)\s*([<>=])\s*(\d+(?:\.\d+)?)',
            r'(DIVIDEND|DIV|YIELD)\s*([<>=])\s*(\d+(?:\.\d+)?)',
            r'(MARKET\s*CAP|MCAP)\s*([<>=])\s*(\d+(?:\.\d+)?)',
            r'(RSI)\s*([<>=])\s*(\d+(?:\.\d+)?)',
            r'(PRICE)\s*([<>=])\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in criteria_patterns:
            match = re.search(pattern, text)
            if match:
                indicator = match.group(1)
                operator = match.group(2)
                value = float(match.group(3))
                
                return {
                    'action': 'screen',
                    'type': 'custom',
                    'indicator': indicator,
                    'operator': operator,
                    'value': value,
                    'original_text': text,
                    'symbols': []  # 添加空的symbols列表
                }
        
        # Screen for predefined categories
        if any(word in text for word in ['DIVIDEND', 'DIV', 'HIGH DIVIDEND']):
            return {
                'action': 'screen',
                'type': 'dividend',
                'original_text': text,
                'symbols': []
            }
        elif any(word in text for word in ['GROWTH', 'HIGH GROWTH']):
            return {
                'action': 'screen',
                'type': 'growth',
                'original_text': text,
                'symbols': []
            }
        elif any(word in text for word in ['VALUE', 'UNDERVALUED', 'CHEAP']):
            return {
                'action': 'screen',
                'type': 'value',
                'original_text': text,
                'symbols': []
            }
        elif any(word in text for word in ['BLUE CHIP', 'LARGE CAP', 'BIG CAP']):
            return {
                'action': 'screen',
                'type': 'blue_chip',
                'original_text': text,
                'symbols': []
            }
        elif any(word in text for word in ['SMALL CAP', 'SMALLCAP']):
            return {
                'action': 'screen',
                'type': 'small_cap',
                'original_text': text,
                'symbols': []
            }
        else:
            return {
                'action': 'screen',
                'type': 'all',
                'original_text': text,
                'symbols': []
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
    
    def process_command(self, text: str) -> Generator[Dict, None, None]:
        """Process user command and route to appropriate analysis"""
        # Parse the command
        parsed = self.parser.parse_command(text)
        
        yield {
            "type": "status",
            "content": f"🔍 Processing command: {parsed['original_text']}"
        }
        
        # 检查是否是屏幕命令
        if parsed['action'] == 'screen':
            yield from self.stream_screening(parsed)
            return
        
        # 检查普通分析命令是否有有效符号
        if not parsed.get('symbols'):
            yield {
                "type": "error",
                "content": "❌ No valid stock symbols found. Please enter symbols like: AAPL, GOOGL, TSLA, NVDA"
            }
            return
        
        # Validate symbols
        valid_symbols = []
        for symbol in parsed['symbols']:
            if self.parser.validate_symbol(symbol):
                valid_symbols.append(symbol)
            else:
                yield {
                    "type": "status",
                    "content": f"⚠️ Skipping invalid symbol: {symbol}"
                }
        
        if not valid_symbols:
            yield {
                "type": "error", 
                "content": "❌ No valid stock symbols found. Examples: AAPL, GOOGL, TSLA, NVDA"
            }
            return
        
        # Route to appropriate analysis
        if parsed['action'] == 'compare' and len(valid_symbols) >= 2:
            yield from self.stream_comparison(valid_symbols[:2])
        else:
            # Single symbol analysis
            yield from self.stream_analysis(valid_symbols[0])
    
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
    
    def stream_screening(self, criteria: Dict) -> Generator[Dict, None, None]:
        """Stream stock screening results"""
        yield {
            "type": "status",
            "content": f"🔍 Starting stock screening: {criteria.get('original_text', 'Custom criteria')}"
        }
        
        # Extended US stock universe
        stock_universe = [
            # FAANG + Tech Giants
            "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE",
            "CRM", "ORCL", "IBM", "INTC", "AMD", "QCOM", "TXN", "NOW", "INTU", "MU",
            
            # Large Cap Blue Chips
            "BRK-B", "UNH", "JNJ", "V", "JPM", "PG", "HD", "MA", "BAC", "ABBV",
            "PFE", "KO", "PEP", "MRK", "COST", "WMT", "DIS", "VZ", "CMCSA", "XOM",
            
            # Financial Sector
            "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI", "USB", "TFC", "PNC",
            "COF", "SCHW", "CB", "MMC", "ICE", "CME", "AON", "TRV", "ALL", "PGR",
            
            # Healthcare & Pharma
            "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "MDT", "BMY",
            "AMGN", "GILD", "CVS", "CI", "ANTM", "HUM", "CNC", "MOH", "ELV", "VEEV",
            
            # Consumer & Retail
            "WMT", "HD", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX",
            "COST", "CVX", "PM", "UL", "CL", "KMB", "GIS", "K", "HSY", "CLX",
            
            # Industrial & Manufacturing
            "BA", "CAT", "HON", "UPS", "GE", "LMT", "MMM", "RTX", "DE", "FDX",
            "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "XYL", "IR", "OTIS", "CARR",
            
            # Energy & Utilities
            "XOM", "CVX", "COP", "EOG", "SLB", "KMI", "OXY", "VLO", "PSX", "MPC",
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ES",
            
            # REITs & Real Estate
            "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "EXR", "AVB", "EQR", "SPG",
            "O", "STOR", "NNN", "ADC", "FRT", "BXP", "VTR", "HCP", "UDR", "ESS",
            
            # Communication Services
            "T", "VZ", "TMUS", "CHTR", "CMCSA", "DIS", "NFLX", "GOOGL", "META", "SNAP",
            
            # Materials & Mining
            "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "GOLD", "AEM", "KGC", "AU",
            
            # Small/Mid Cap Growth
            "ROKU", "ZM", "DOCU", "SNOW", "PLTR", "RBLX", "U", "NET", "CRWD", "ZS",
            "OKTA", "TWLO", "DDOG", "MDB", "ESTC", "FSLY", "CFLT", "S", "WORK", "TEAM"
        ]
        
        yield {
            "type": "status",
            "content": f"📊 Screening {len(stock_universe)} stocks..."
        }
        
        results = []
        processed = 0
        
        for symbol in stock_universe:
            try:
                processed += 1
                if processed % 20 == 0:  # Update progress every 20 stocks
                    yield {
                        "type": "status",
                        "content": f"🔄 Progress: {processed}/{len(stock_universe)} stocks analyzed..."
                    }
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                info = ticker.info
                
                if data.empty:
                    continue
                
                # Quick analysis
                technical_data = self._calculate_technical_indicators(data)
                fundamental_data = self._extract_fundamentals(info)
                
                stock_data = {
                    'symbol': symbol,
                    'name': info.get('longName', symbol)[:30],
                    'sector': info.get('sector', 'Unknown')[:20],
                    'price': data['Close'].iloc[-1],
                    'change': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0,
                    'pe_ratio': fundamental_data.get('pe_ratio'),
                    'pb_ratio': fundamental_data.get('pb_ratio'),
                    'roe': fundamental_data.get('roe'),
                    'debt_to_equity': fundamental_data.get('debt_to_equity'),
                    'dividend_yield': fundamental_data.get('dividend_yield'),
                    'market_cap': fundamental_data.get('market_cap'),
                    'rsi': technical_data.get('rsi'),
                    'revenue_growth': fundamental_data.get('revenue_growth'),
                    'beta': fundamental_data.get('beta')
                }
                
                # Apply screening criteria
                if self._meets_criteria(stock_data, criteria):
                    results.append(stock_data)
                    
            except Exception:
                continue  # Skip problematic stocks
        
        # Sort results based on criteria
        results = self._sort_screening_results(results, criteria)
        
        yield {
            "type": "status",
            "content": f"✅ Screening complete! Found {len(results)} matching stocks"
        }
        
        yield {
            "type": "screening",
            "content": {
                'criteria': criteria,
                'results': results,
                'total_screened': len(stock_universe),
                'matches_found': len(results)
            }
        }
        
        yield {
            "type": "complete",
            "content": f"🎯 Screening complete: {len(results)} stocks match your criteria"
        }
    
    def _meets_criteria(self, stock_data: Dict, criteria: Dict) -> bool:
        """Check if stock meets screening criteria"""
        criteria_type = criteria.get('type', 'all')
        
        if criteria_type == 'custom':
            indicator = criteria.get('indicator', '').upper()
            operator = criteria.get('operator', '>')
            value = criteria.get('value', 0)
            
            if indicator in ['PE', 'P/E']:
                pe = stock_data.get('pe_ratio')
                if pe is None or pe <= 0:
                    return False
                if operator == '<': return pe < value
                elif operator == '>': return pe > value
                elif operator == '=': return abs(pe - value) < 0.5
                
            elif indicator in ['PB', 'P/B']:
                pb = stock_data.get('pb_ratio')
                if pb is None or pb <= 0:
                    return False
                if operator == '<': return pb < value
                elif operator == '>': return pb > value
                elif operator == '=': return abs(pb - value) < 0.1
                
            elif indicator == 'ROE':
                roe = stock_data.get('roe')
                if roe is None:
                    return False
                roe_percent = roe * 100
                if operator == '<': return roe_percent < value
                elif operator == '>': return roe_percent > value
                elif operator == '=': return abs(roe_percent - value) < 1
                
            elif indicator in ['DEBT', 'DEBT/EQUITY', 'DEBT-TO-EQUITY']:
                debt_eq = stock_data.get('debt_to_equity')
                if debt_eq is None:
                    return False
                if operator == '<': return debt_eq < value
                elif operator == '>': return debt_eq > value
                elif operator == '=': return abs(debt_eq - value) < 0.1
                
            elif indicator in ['DIVIDEND', 'DIV', 'YIELD']:
                div_yield = stock_data.get('dividend_yield')
                if div_yield is None:
                    return False
                div_percent = div_yield * 100
                if operator == '<': return div_percent < value
                elif operator == '>': return div_percent > value
                elif operator == '=': return abs(div_percent - value) < 0.1
                
            elif indicator == 'RSI':
                rsi = stock_data.get('rsi')
                if rsi is None:
                    return False
                if operator == '<': return rsi < value
                elif operator == '>': return rsi > value
                elif operator == '=': return abs(rsi - value) < 2
                
            elif indicator == 'PRICE':
                price = stock_data.get('price')
                if price is None:
                    return False
                if operator == '<': return price < value
                elif operator == '>': return price > value
                elif operator == '=': return abs(price - value) < 1
        
        elif criteria_type == 'dividend':
            div_yield = stock_data.get('dividend_yield')
            return div_yield and div_yield > 0.03  # > 3%
            
        elif criteria_type == 'growth':
            revenue_growth = stock_data.get('revenue_growth')
            roe = stock_data.get('roe')
            return (revenue_growth and revenue_growth > 0.15) or (roe and roe > 0.2)
            
        elif criteria_type == 'value':
            pe = stock_data.get('pe_ratio')
            pb = stock_data.get('pb_ratio')
            return (pe and 5 < pe < 20) and (pb and pb < 3)
            
        elif criteria_type == 'blue_chip':
            market_cap = stock_data.get('market_cap')
            return market_cap and market_cap > 10e9  # > $10B
            
        elif criteria_type == 'small_cap':
            market_cap = stock_data.get('market_cap')
            return market_cap and market_cap < 2e9  # < $2B
        
        return True  # 'all' type
    
    def _sort_screening_results(self, results: List[Dict], criteria: Dict) -> List[Dict]:
        """Sort screening results based on criteria"""
        criteria_type = criteria.get('type', 'all')
        
        if criteria_type == 'dividend':
            return sorted(results, key=lambda x: x.get('dividend_yield', 0) or 0, reverse=True)
        elif criteria_type == 'growth':
            return sorted(results, key=lambda x: x.get('revenue_growth', 0) or 0, reverse=True)
        elif criteria_type == 'value':
            return sorted(results, key=lambda x: x.get('pe_ratio', 999) or 999)
        elif criteria_type == 'blue_chip' or criteria_type == 'small_cap':
            return sorted(results, key=lambda x: x.get('market_cap', 0) or 0, reverse=True)
        else:
            # Custom or default sort by market cap
            return sorted(results, key=lambda x: x.get('market_cap', 0) or 0, reverse=True)
        
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

# Apply selected theme
st.markdown(get_theme_css(theme), unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "analyzer" not in st.session_state:
    st.session_state.analyzer = StreamingAnalyzer()

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
        if st.button("🔍 Screen PE < 20", key="q2"):
            st.session_state.messages.append({"role": "user", "content": "screen PE < 20"})
            st.rerun()
    
    with col3:
        if st.button("⚖️ Compare NVDA vs AAPL", key="q3"):
            st.session_state.messages.append({"role": "user", "content": "compare NVDA and AAPL"})
            st.rerun()
    
    with col4:
        if st.button("💰 Find Dividend Stocks", key="q4"):
            st.session_state.messages.append({"role": "user", "content": "find dividend stocks"})
            st.rerun()
    
    # Advanced Examples Section
    st.markdown("### 🎯 Advanced Screening Examples")
    
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        st.markdown("**📈 Technical Screening:**")
        if st.button("🔴 RSI Oversold (< 30)", key="adv1"):
            st.session_state.messages.append({"role": "user", "content": "screen RSI < 30"})
            st.rerun()
        if st.button("💎 Value Stocks (PE < 15)", key="adv2"):
            st.session_state.messages.append({"role": "user", "content": "screen PE < 15"})
            st.rerun()
        if st.button("🏦 Low Debt (< 0.3)", key="adv3"):
            st.session_state.messages.append({"role": "user", "content": "screen debt < 0.3"})
            st.rerun()
    
    with adv_col2:
        st.markdown("**🎲 Category Screening:**")
        if st.button("🌟 Growth Stocks", key="adv4"):
            st.session_state.messages.append({"role": "user", "content": "find growth stocks"})
            st.rerun()
        if st.button("🔷 Blue Chip Stocks", key="adv5"):
            st.session_state.messages.append({"role": "user", "content": "find blue chip stocks"})
            st.rerun()
        if st.button("🚀 Small Cap Stocks", key="adv6"):
            st.session_state.messages.append({"role": "user", "content": "find small cap stocks"})
            st.rerun()
    
    # Command Examples
    st.markdown("### 📚 Command Examples")
    
    with st.expander("🔍 Screening Commands", expanded=True):
        st.markdown("""
        **Basic Screening:**
        - `screen PE < 20` - Find stocks with P/E ratio below 20
        - `screen PB < 2` - Find stocks with P/B ratio below 2
        - `screen ROE > 15` - Find stocks with ROE above 15%
        - `screen dividend > 4` - Find stocks with dividend yield above 4%
        
        **Technical Screening:**
        - `screen RSI < 30` - Find oversold stocks
        - `screen RSI > 70` - Find overbought stocks
        - `screen price < 50` - Find stocks under $50
        
        **Category Screening:**
        - `find dividend stocks` - High dividend yield stocks
        - `find value stocks` - Undervalued stocks
        - `find growth stocks` - High growth potential
        - `find blue chip stocks` - Large cap stable companies
        """)
    
    with st.expander("📊 Analysis Commands"):
        st.markdown("""
        **Single Stock Analysis:**
        - `AAPL` or `analyze AAPL` - Comprehensive analysis
        - `check NVDA` - Quick check
        - `look at TSLA` - Detailed view
        
        **Stock Comparison:**
        - `compare AAPL and GOOGL` - Side by side comparison
        - `NVDA vs TSLA` - Quick comparison
        - `MSFT versus AMZN` - Detailed comparison
        """)
        
    with st.expander("🎯 Pro Tips"):
        st.markdown("""
        **Advanced Usage:**
        - Use operators: `<`, `>`, `=` for precise screening
        - Combine criteria: Screen first, then analyze individual stocks
        - Compare similar stocks: Find stocks in same sector, then compare
        
        **Best Practices:**
        - Start with broad screening, then narrow down
        - Always verify with fundamental analysis
        - Consider multiple timeframes for technical analysis
        - Use comparison for final investment decisions
        """)
    
    # Market Insights
    st.markdown("### 📈 Market Quick Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">📊</div>
            <div class="metric-label">150+ STOCKS TRACKED</div>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🔍</div>
            <div class="metric-label">8+ SCREENING CRITERIA</div>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">⚡</div>
            <div class="metric-label">REAL-TIME ANALYSIS</div>
        </div>
        """, unsafe_allow_html=True)

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

# Input section
st.markdown("### 💬 Terminal Input")

col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    user_input = st.text_input(
        "Command",
        placeholder="Examples: AAPL, compare NVDA and AAPL, screen PE < 20, find dividend stocks...",
        key="terminal_input",
        label_visibility="collapsed"
    )

with col2:
    execute_btn = st.button("🚀 EXECUTE", type="primary", use_container_width=True)

with col3:
    if st.session_state.messages:
        clear_btn = st.button("🗑️ CLEAR", use_container_width=True)
        if clear_btn:
            st.session_state.messages = []
            st.rerun()

# Process input with streaming
if execute_btn and user_input.strip():
    # Parse command to extract symbols and action
    parsed_command = st.session_state.analyzer.parser.parse_command(user_input.strip())
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    
    # Create streaming container
    streaming_container = st.empty()
    result_content = ""
    chart_data = None
    technical_table = None
    fundamental_table = None
    comparison_table = None
    
    # Stream the analysis
    for update in st.session_state.analyzer.process_command(user_input.strip()):
        if update["type"] == "status":
            streaming_container.markdown(f"""
            <div class="streaming-message">
                <strong>🤖 TERMINAL:</strong><br/>
                {update["content"]}
            </div>
            """, unsafe_allow_html=True)
            
        elif update["type"] == "info":
            info = update["content"]
            price_color = "#10B981" if info["change"] > 0 else "#EF4444"
            result_content += f"""
## 📈 {info["symbol"]} - {info["name"]}

**Sector:** {info["sector"]} | **Price:** ${info["price"]:.2f} | **Change:** <span style="color: {price_color}">{info["change"]:+.2f}%</span>

"""
            
        elif update["type"] == "technical":
            tech = update["content"]
            result_content += """
### 🔬 Technical Analysis

"""
            # Create technical indicators table
            tech_data = []
            if tech.get('rsi'): tech_data.append(["RSI (14)", f"{tech['rsi']:.1f}"])
            if tech.get('sma_20'): tech_data.append(["SMA 20", f"${tech['sma_20']:.2f}"])
            if tech.get('sma_50'): tech_data.append(["SMA 50", f"${tech['sma_50']:.2f}"])
            if tech.get('macd'): tech_data.append(["MACD", f"{tech['macd']:.4f}"])
            if tech.get('volume_ratio'): tech_data.append(["Volume Ratio", f"{tech['volume_ratio']:.2f}x"])
            
            if tech_data:
                technical_table = pd.DataFrame(tech_data, columns=["Indicator", "Value"])
            
        elif update["type"] == "fundamental":
            fund = update["content"]
            result_content += """
### 💰 Fundamental Analysis

"""
            # Create fundamental table
            fund_data = []
            if fund.get('pe_ratio'): fund_data.append(["P/E Ratio", f"{fund['pe_ratio']:.2f}"])
            if fund.get('pb_ratio'): fund_data.append(["P/B Ratio", f"{fund['pb_ratio']:.2f}"])
            if fund.get('roe'): fund_data.append(["ROE", f"{fund['roe']*100:.1f}%"])
            if fund.get('debt_to_equity'): fund_data.append(["Debt/Equity", f"{fund['debt_to_equity']:.2f}"])
            if fund.get('dividend_yield'): fund_data.append(["Dividend Yield", f"{fund['dividend_yield']*100:.2f}%"])
            if fund.get('market_cap'): fund_data.append(["Market Cap", f"${fund['market_cap']/1e9:.1f}B"])
            
            if fund_data:
                fundamental_table = pd.DataFrame(fund_data, columns=["Metric", "Value"])
            
        elif update["type"] == "insights":
            insights = update["content"]
            result_content += f"""
### 🎯 AI Insights

**Recommendation:** {insights["recommendation"]} | **Confidence:** {insights["confidence"]:.1%} | **Risk Level:** {insights["risk_level"]}

"""
            if insights.get("target_price"):
                result_content += f"**Target Price:** ${insights['target_price']:.2f
