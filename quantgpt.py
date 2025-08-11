# quantgpt.py
# QuantGPT - AI-Powered Trading Assistant

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

# ===================================
# Page Configuration
# ===================================
st.set_page_config(
    page_title="QuantGPT - AI Trading Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================================
# Custom CSS
# ===================================
st.markdown("""
<style>
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 10px;
        padding: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f0f4f8;
    }
    div[data-testid="stMetricValue"] {
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ===================================
# QuantGPT Assistant Class
# ===================================

class QuantGPTAssistant:
    """QuantGPT AI Assistant"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": """👋 Hello! I'm QuantGPT, your AI-powered trading assistant!

I can help you with:
- 📊 **Stock Analysis** (e.g., "analyze AAPL")
- 📈 **Technical Indicators** (e.g., "what's TSLA's RSI?")
- 💡 **Investment Advice** (e.g., "should I buy NVDA?")
- 📉 **Stock Comparison** (e.g., "compare AAPL and GOOGL")
- 🎯 **Strategy Backtesting** (e.g., "test moving average strategy for MSFT")
- 💰 **Return Calculation** (e.g., "if I invest $10,000 in TSLA?")

Feel free to ask me anything about stocks and investing!"""}
            ]
        
        if 'analyzing' not in st.session_state:
            st.session_state.analyzing = False
    
    @st.cache_data(ttl=3600)
    def get_stock_data(_self, symbol, period="1y"):
        """Fetch stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            info = ticker.info
            return data, info
        except:
            # Return mock data as fallback
            return _self.generate_mock_data(symbol, period)
    
    def generate_mock_data(self, symbol, period="1y"):
        """Generate mock data for demonstration"""
        periods_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
        days = periods_days.get(period, 365)
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(hash(symbol) % 10000)
        
        base_price = {"AAPL": 180, "GOOGL": 140, "MSFT": 380, "TSLA": 250, "NVDA": 500}.get(symbol, 100)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = data['Close'] * np.random.uniform(0.98, 1.02, len(dates))
        data['High'] = np.maximum(data['Open'], data['Close']) * np.random.uniform(1.0, 1.02, len(dates))
        data['Low'] = np.minimum(data['Open'], data['Close']) * np.random.uniform(0.98, 1.0, len(dates))
        data['Volume'] = np.random.randint(10000000, 100000000, len(dates))
        
        info = {
            'longName': f'{symbol} Inc.',
            'sector': 'Technology',
            'marketCap': base_price * 1000000000,
            'trailingPE': np.random.uniform(15, 35)
        }
        
        return data, info
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        df = data.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        return df
    
    def analyze_stock(self, symbol):
        """Analyze a stock"""
        data, info = self.get_stock_data(symbol)
        data = self.calculate_indicators(data)
        
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        change = (current_price - prev_close) / prev_close * 100
        
        rsi = data['RSI'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        
        # Generate recommendation
        score = 50
        if current_price > sma_20 > sma_50:
            score += 20
        if 30 < rsi < 70:
            score += 10
        elif rsi < 30:
            score += 15
        
        if score >= 70:
            recommendation = "🟢 **BUY**"
            reason = "Strong bullish signals detected"
        elif score >= 60:
            recommendation = "🟡 **HOLD**"
            reason = "Mixed signals, wait for clearer direction"
        else:
            recommendation = "🔴 **SELL**"
            reason = "Bearish indicators suggest caution"
        
        analysis = f"""
### 📊 {symbol} Analysis Report

**Basic Information:**
- Current Price: ${current_price:.2f}
- Daily Change: {change:+.2f}%
- Volume: {data['Volume'].iloc[-1]:,.0f}

**Technical Indicators:**
- RSI(14): {rsi:.2f} {'(Oversold)' if rsi < 30 else '(Overbought)' if rsi > 70 else '(Neutral)'}
- SMA20: ${sma_20:.2f}
- SMA50: ${sma_50:.2f}
- Trend: {'Bullish 📈' if current_price > sma_20 > sma_50 else 'Bearish 📉' if current_price < sma_20 < sma_50 else 'Sideways 📊'}

**AI Recommendation:** {recommendation}
**Reasoning:** {reason}

**Risk Warning:** All investments carry risk. Please do your own research.
"""
        return analysis, data
    
    def create_chart(self, symbol, data):
        """Create price chart"""
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        # Add moving averages
        if 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, 
                y=data['SMA_20'],
                name='SMA20',
                line=dict(color='orange', width=1)
            ))
        
        if 'SMA_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                name='SMA50',
                line=dict(color='blue', width=1)
            ))
        
        fig.update_layout(
            title=f'{symbol} Price Chart',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            height=400,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def process_query(self, query):
        """Process user query"""
        query_lower = query.lower()
        
        # Extract stock symbols
        import re
        stock_symbols = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        
        # Analysis queries
        if any(word in query_lower for word in ['analyze', 'analysis', 'check', 'look at', 'review']):
            if stock_symbols:
                symbol = stock_symbols[0]
                analysis, data = self.analyze_stock(symbol)
                chart = self.create_chart(symbol, data)
                return analysis, chart
            else:
                return "Please provide a stock symbol. For example: 'analyze AAPL'", None
        
        # Price queries
        elif any(word in query_lower for word in ['price', 'cost', 'worth', 'trading at']):
            if stock_symbols:
                symbol = stock_symbols[0]
                data, _ = self.get_stock_data(symbol, "1mo")
                price = data['Close'].iloc[-1]
                change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                return f"**{symbol}** is currently trading at **${price:.2f}** ({change:+.2f}%)", None
            else:
                return "Please provide a stock symbol. For example: 'What's AAPL's price?'", None
        
        # Buy/Sell advice
        elif any(word in query_lower for word in ['buy', 'sell', 'should i', 'recommend', 'suggestion']):
            if stock_symbols:
                symbol = stock_symbols[0]
                analysis, data = self.analyze_stock(symbol)
                return analysis, None
            else:
                # Recommend popular stocks
                return """### 🔥 Today's Top Picks:

1. **NVDA** - AI chip leader, strong momentum 🟢
2. **AAPL** - Stable blue-chip, good for long-term 🟢
3. **TSLA** - EV leader, high volatility 🟡
4. **MSFT** - Cloud giant, steady growth 🟢
5. **GOOGL** - Search dominance, AI transformation 🟡

Enter a specific ticker for detailed analysis!""", None
        
        # RSI queries
        elif 'rsi' in query_lower:
            if stock_symbols:
                symbol = stock_symbols[0]
                data, _ = self.get_stock_data(symbol)
                data = self.calculate_indicators(data)
                rsi = data['RSI'].iloc[-1]
                
                if rsi < 30:
                    status = "**Oversold** 🟢 - Potential buying opportunity"
                elif rsi > 70:
                    status = "**Overbought** 🔴 - May face correction"
                else:
                    status = "**Neutral** 🟡 - No clear signal"
                
                return f"**{symbol}** RSI(14) = **{rsi:.2f}**\n\n{status}", None
        
        # Compare stocks
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'better']):
            if len(stock_symbols) >= 2:
                results = []
                returns = []
                for symbol in stock_symbols[:2]:
                    data, _ = self.get_stock_data(symbol, "1mo")
                    monthly_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
                    results.append(f"**{symbol}**: Monthly return {monthly_return:+.2f}%")
                    returns.append(monthly_return)
                
                winner = stock_symbols[0] if returns[0] > returns[1] else stock_symbols[1]
                return f"### 📊 Stock Comparison\n\n" + "\n".join(results) + f"\n\n🏆 **{winner}** is performing better!", None
            else:
                return "Please provide two stock symbols to compare. For example: 'compare AAPL and GOOGL'", None
        
        # Investment calculation
        elif any(word in query_lower for word in ['invest', 'return', 'profit', 'if i', 'calculate']):
            if stock_symbols and any(char.isdigit() for char in query):
                # Extract amount
                amounts = re.findall(r'[\d,]+', query)
                if amounts:
                    amount = float(amounts[0].replace(',', ''))
                    symbol = stock_symbols[0]
                    data, _ = self.get_stock_data(symbol, "1y")
                    
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    shares = amount / start_price
                    final_value = shares * end_price
                    profit = final_value - amount
                    return_rate = (profit / amount) * 100
                    
                    return f"""### 💰 Investment Simulation - {symbol}

**Initial Investment:** ${amount:,.2f}
**Entry Price:** ${start_price:.2f}
**Current Price:** ${end_price:.2f}
**Shares:** {shares:.2f}

**Current Value:** ${final_value:,.2f}
**Profit/Loss:** ${profit:+,.2f}
**Return:** {return_rate:+.2f}%

{'🎉 Congratulations! Your investment is profitable!' if profit > 0 else '😔 Currently at a loss, consider holding for recovery'}""", None
        
        # Moving average strategy
        elif any(word in query_lower for word in ['strategy', 'backtest', 'test', 'moving average', 'ma']):
            if stock_symbols:
                symbol = stock_symbols[0]
                data, _ = self.get_stock_data(symbol)
                data = self.calculate_indicators(data)
                
                # Simple MA crossover strategy
                buy_signals = 0
                sell_signals = 0
                
                for i in range(50, len(data)):
                    if data['SMA_20'].iloc[i] > data['SMA_50'].iloc[i] and data['SMA_20'].iloc[i-1] <= data['SMA_50'].iloc[i-1]:
                        buy_signals += 1
                    elif data['SMA_20'].iloc[i] < data['SMA_50'].iloc[i] and data['SMA_20'].iloc[i-1] >= data['SMA_50'].iloc[i-1]:
                        sell_signals += 1
                
                return f"""### 📈 Moving Average Strategy - {symbol}

**Strategy:** SMA20/SMA50 Crossover
**Period:** Last 12 months

**Signals Generated:**
- Buy Signals: {buy_signals}
- Sell Signals: {sell_signals}

**Current Status:** {'Bullish' if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] else 'Bearish'}

This is a classic trend-following strategy that works well in trending markets.""", None
        
        # Default response
        else:
            return """I can help you with:

- 📊 **Stock Analysis**: Type "analyze AAPL"
- 💵 **Price Check**: Type "TSLA price"
- 📈 **Technical Indicators**: Type "NVDA RSI"
- 🔄 **Compare Stocks**: Type "compare AAPL and GOOGL"
- 💰 **Investment Simulation**: Type "invest $10000 in MSFT"
- 🎯 **Strategy Testing**: Type "test strategy for AAPL"
- 💡 **Get Recommendations**: Type "what should I buy?"

What would you like to know?""", None

# ===================================
# Main Application
# ===================================

def main():
    st.title("🤖 QuantGPT - AI Trading Assistant")
    st.markdown("---")
    
    # Initialize assistant
    assistant = QuantGPTAssistant()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📌 Quick Actions")
        
        # Popular stocks buttons
        st.markdown("**Popular Stocks:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🍎 AAPL"):
                st.session_state.messages.append({"role": "user", "content": "analyze AAPL"})
            if st.button("🚗 TSLA"):
                st.session_state.messages.append({"role": "user", "content": "analyze TSLA"})
            if st.button("🖥️ NVDA"):
                st.session_state.messages.append({"role": "user", "content": "analyze NVDA"})
        with col2:
            if st.button("🔍 GOOGL"):
                st.session_state.messages.append({"role": "user", "content": "analyze GOOGL"})
            if st.button("💻 MSFT"):
                st.session_state.messages.append({"role": "user", "content": "analyze MSFT"})
            if st.button("📱 META"):
                st.session_state.messages.append({"role": "user", "content": "analyze META"})
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What's AAPL's price?",
            "analyze TSLA",
            "What's NVDA's RSI?",
            "compare AAPL and GOOGL",
            "if I invest $10000 in MSFT?",
            "what should I buy today?"
        ]
        
        for question in example_questions:
            if st.button(f"📝 {question}", key=question):
                st.session_state.messages.append({"role": "user", "content": question})
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display chart if available
            if "chart" in message and message["chart"] is not None:
                st.plotly_chart(message["chart"], use_container_width=True)
    
    # User input
    if prompt := st.chat_input("Ask me anything about stocks..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, chart = assistant.process_query(prompt)
                st.markdown(response)
                
                # Display chart if available
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Save assistant message
                message_data = {"role": "assistant", "content": response}
                if chart is not None:
                    message_data["chart"] = chart
                st.session_state.messages.append(message_data)

if __name__ == "__main__":
    main()
