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

class StreamingAnalyzer:
    """Streaming Analysis Engine"""
    
    def __init__(self):
        self.data_cache = {}
    
    def stream_analysis(self, symbol: str) -> Generator[Dict, None, None]:
        """Stream analysis results progressively"""
        
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
                    "content": f"❌ No data found for symbol: {symbol.upper()}"
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
                "content": f"❌ Analysis failed: {str(e)}"
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
        if st.button("🔍 Analyze GOOGL", key="q2"):
            st.session_state.messages.append({"role": "user", "content": "GOOGL"})
            st.rerun()
    
    with col3:
        if st.button("⚡ Analyze TSLA", key="q3"):
            st.session_state.messages.append({"role": "user", "content": "TSLA"})
            st.rerun()
    
    with col4:
        if st.button("💎 Analyze NVDA", key="q4"):
            st.session_state.messages.append({"role": "user", "content": "NVDA"})
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

# Input section
st.markdown("### 💬 Terminal Input")

col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    user_input = st.text_input(
        "Command",
        placeholder="Enter stock symbol (e.g., AAPL, GOOGL, TSLA, NVDA...)",
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
    symbol = user_input.strip().upper()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": symbol})
    
    # Create streaming container
    streaming_container = st.empty()
    result_content = ""
    chart_data = None
    technical_table = None
    fundamental_table = None
    
    # Stream the analysis
    for update in st.session_state.analyzer.stream_analysis(symbol):
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
                result_content += f"**Target Price:** ${insights['target_price']:.2f}\n\n"
            
            # Add signals
            if insights.get("signals"):
                result_content += "**Key Signals:**\n"
                for signal in insights["signals"]:
                    emoji = "🟢" if signal["type"] == "bullish" else "🔴"
                    result_content += f"- {emoji} {signal['message']}\n"
                result_content += "\n"
            
        elif update["type"] == "chart":
            chart_data = update["content"]
            
        elif update["type"] == "complete":
            # Final update - clear streaming container and add final message
            streaming_container.empty()
            
            final_message = {
                "role": "assistant",
                "content": result_content
            }
            
            if chart_data:
                final_message["chart"] = chart_data
            if technical_table is not None:
                final_message["technical_table"] = technical_table
            if fundamental_table is not None:
                final_message["fundamental_table"] = fundamental_table
            
            st.session_state.messages.append(final_message)
            st.rerun()
            
        elif update["type"] == "error":
            streaming_container.markdown(f"""
            <div class="ai-message">
                <strong>🤖 TERMINAL:</strong><br/>
                {update["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": update["content"]
            })
            time.sleep(2)
            st.rerun()

# Professional Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: var(--text-secondary); padding: 2rem; font-family: "Inter", sans-serif;'>
    <p><strong>📈 US Stock Terminal Pro v1.0</strong></p>
    <p>Real-time Market Analysis • Professional Trading Intelligence • AI-Powered Insights</p>
    <p style="font-size: 0.75rem; margin-top: 1rem;">
        ⚠️ For educational and research purposes only. Not financial advice.
    </p>
</div>
""", unsafe_allow_html=True)
