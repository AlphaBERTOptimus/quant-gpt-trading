import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import re
import time
import hashlib
from typing import Dict, List, Generator, Any

# 修复输入框文本颜色问题
def get_theme_css():
    return """
<style>
    :root {
        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --text-primary: #F8FAFC;
        --accent-color: #3B82F6;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
    }
    .stApp { background: var(--bg-primary); color: var(--text-primary); }
    .terminal-header { background: linear-gradient(135deg, var(--bg-secondary) 0%, #334155 100%); padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; }
    .status-bar { display: flex; gap: 1rem; background: var(--bg-secondary); padding: 0.5rem 1rem; border-radius: 6px; margin-bottom: 1rem; font-size: 0.8rem; }
    .user-message { background: var(--bg-secondary); padding: 1rem; margin: 0.5rem 0; border-radius: 6px; border-left: 3px solid var(--accent-color); }
    .ai-message { background: var(--bg-secondary); padding: 1rem; margin: 0.5rem 0; border-radius: 6px; border-left: 3px solid var(--success-color); }
    
    /* 修复按钮样式 - 确保文字可见 */
    .stButton>button {
        background: var(--accent-color) !important;
        color: white !important;
        border-radius: 6px;
    }
    
    .stButton>button:hover {
        background: var(--success-color) !important;
    }
    
    /* 修复输入框文本颜色 */
    .stTextInput>div>div>input {
        color: var(--text-primary) !important;
    }
</style>
"""

class StockDatabase:
    def get_all_us_stocks(self) -> List[str]:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'HD', 'MA', 'DIS', 'BAC', 'CRM', 'KO', 'PFE', 'INTC', 'VZ', 'WMT', 'XOM', 'CVX']

class CommandParser:
    @staticmethod
    def parse_command(text: str) -> Dict:
        text = text.upper().strip()
        symbols = re.findall(r'\b[A-Z]{1,5}\b', text)
        action = 'analyze'
        params = {}
        
        if 'SCREEN' in text:
            action = 'screen'
            if 'PE' in text:
                if '<' in text:
                    pe_match = re.search(r'PE.*?<.*?(\d+)', text)
                    if pe_match: params['pe_max'] = float(pe_match.group(1))
        elif 'COMPARE' in text or 'VS' in text:
            action = 'compare'
        elif '*' in text:
            action = 'check_all'
            prefix = re.search(r'([A-Z]+)\*', text)
            if prefix: params['prefix'] = prefix.group(1)
        
        return {'action': action, 'symbols': symbols[:5], 'params': params}

class StockAnalyzer:
    def __init__(self):
        self.parser = CommandParser()
        self.stock_db = StockDatabase()
    
    def process_command(self, text: str) -> Generator[Dict, None, None]:
        """Process user command and route to appropriate analysis"""
        parsed = self.parser.parse_command(text)
        yield {"type": "status", "content": f"🔍 Processing command: {text}"}
        
        if parsed['action'] == 'screen':
            yield from self.screen_stocks(parsed['params'])
        elif parsed['action'] == 'compare' and len(parsed['symbols']) >= 2:
            yield from self.compare_stocks(parsed['symbols'])
        elif parsed['action'] == 'check_all':
            yield from self.check_all_stocks(parsed['params'])
        elif parsed['symbols']:
            if len(parsed['symbols']) == 1:
                yield from self.analyze_stock(parsed['symbols'][0])
            else:
                yield from self.analyze_multiple_stocks(parsed['symbols'])
        else:
            yield {"type": "error", "content": "❌ Invalid command. Try: AAPL, screen PE<15, compare AAPL MSFT, TECH*"}
    
    def analyze_stock(self, symbol: str) -> Generator[Dict, None, None]:
        """Analyze a single stock"""
        yield {"type": "status", "content": f"📊 Analyzing {symbol}..."}
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            info = ticker.info
            
            if data.empty:
                yield {"type": "error", "content": f"⚠️ No data found for {symbol}"}
                return
            
            # Calculate key metrics
            price = data['Close'].iloc[-1]
            change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
            pe = info.get('trailingPE', 'N/A')
            market_cap = f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A'
            
            # Calculate RSI
            rsi = self.calculate_rsi(data) if len(data) >= 14 else 'N/A'
            
            # Create chart
            fig = self.create_chart(data)
            
            yield {
                "type": "analysis",
                "content": {
                    'symbol': symbol,
                    'price': price,
                    'change': change,
                    'pe': pe,
                    'market_cap': market_cap,
                    'rsi': rsi,
                    'chart': fig
                }
            }
            
        except Exception as e:
            yield {"type": "error", "content": f"❌ Failed to analyze {symbol}: {str(e)}"}
    
    def analyze_multiple_stocks(self, symbols: List[str]) -> Generator[Dict, None, None]:
        """Analyze multiple stocks"""
        yield {"type": "status", "content": f"📊 Analyzing {len(symbols)} stocks: {', '.join(symbols)}"}
        
        results = []
        for i, symbol in enumerate(symbols):
            yield {"type": "status", "content": f"🔄 Analyzing {symbol} ({i+1}/{len(symbols)})..."}
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                info = ticker.info
                
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
                    pe = info.get('trailingPE', 'N/A')
                    market_cap = f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A'
                    
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol)[:40],
                        'price': price,
                        'change': change,
                        'pe': pe,
                        'market_cap': market_cap
                    })
            except Exception as e:
                yield {"type": "status", "content": f"⚠️ Could not analyze {symbol}: {str(e)}"}
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
            yield {"type": "error", "content": "❌ Could not analyze any of the provided symbols"}
    
    def calculate_rsi(self, data: pd.DataFrame) -> float:
        """Calculate RSI indicator"""
        close = data['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).iloc[-1]
    
    def create_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create stock price chart"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        # Price chart
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
        
        # Volume chart
        colors = ['green' if close > open else 'red' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors, opacity=0.7),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Stock Price Chart",
            height=600,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def screen_stocks(self, criteria: Dict) -> Generator[Dict, None, None]:
        """Screen stocks based on criteria"""
        yield {"type": "status", "content": "🔍 Screening stocks based on your criteria..."}
        
        # Get stock universe
        stocks = self.stock_db.get_all_us_stocks()[:20]  # Limit for demo
        
        yield {"type": "status", "content": f"📊 Analyzing {len(stocks)} stocks..."}
        
        results = []
        for i, symbol in enumerate(stocks):
            if (i + 1) % 5 == 0:
                yield {"type": "status", "content": f"🔄 Processed {i+1}/{len(stocks)} stocks..."}
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")
                info = ticker.info
                
                if not data.empty:
                    pe = info.get('trailingPE', 1000)
                    
                    # Check criteria
                    if 'pe_max' in criteria and pe > criteria['pe_max']:
                        continue
                        
                    results.append({
                        'symbol': symbol,
                        'price': data['Close'].iloc[-1],
                        'pe': pe,
                        'market_cap': info.get('marketCap')
                    })
            except:
                continue
        
        if results:
            # Sort by PE ratio
            results.sort(key=lambda x: x.get('pe', 999))
            
            yield {
                "type": "screening",
                "content": {
                    'results': results[:10],  # Top 10 results
                    'criteria': criteria
                }
            }
        else:
            yield {"type": "error", "content": "❌ No stocks found matching your criteria"}
    
    def compare_stocks(self, symbols: List[str]) -> Generator[Dict, None, None]:
        """Compare multiple stocks"""
        yield {"type": "status", "content": f"⚖️ Comparing {len(symbols)} stocks: {', '.join(symbols)}"}
        
        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                info = ticker.info
                
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100)
                    results.append({
                        'symbol': symbol,
                        'price': price,
                        'ytd_change': change,
                        'pe': info.get('trailingPE', 'N/A')
                    })
            except:
                continue
        
        if results:
            yield {
                "type": "comparison",
                "content": {
                    'results': results,
                    'symbols': symbols
                }
            }
        else:
            yield {"type": "error", "content": "❌ Could not compare the provided stocks"}
    
    def check_all_stocks(self, params: Dict) -> Generator[Dict, None, None]:
        """Find all stocks matching a pattern"""
        prefix = params.get('prefix', '')
        yield {"type": "status", "content": f"🔍 Finding all stocks starting with '{prefix}'..."}
        
        # Get stock universe
        all_stocks = self.stock_db.get_all_us_stocks()
        matching = [s for s in all_stocks if s.startswith(prefix)][:10]  # Limit to 10
        
        if not matching:
            yield {"type": "error", "content": f"❌ No stocks found starting with '{prefix}'"}
            return
        
        yield {"type": "status", "content": f"📊 Found {len(matching)} matching stocks. Analyzing..."}
        
        results = []
        for symbol in matching:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")
                info = ticker.info
                
                if not data.empty:
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol)[:40],
                        'price': data['Close'].iloc[-1],
                        'volume': data['Volume'].iloc[-1]
                    })
            except:
                continue
        
        if results:
            yield {
                "type": "check_all",
                "content": {
                    'results': results,
                    'prefix': prefix
                }
            }
        else:
            yield {"type": "error", "content": f"❌ Could not analyze stocks starting with '{prefix}'"}

# Streamlit界面
def main():
    st.set_page_config(page_title="Stock Terminal Pro", layout="wide")
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
    # 初始化session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    # 初始化输入状态
    if "input" not in st.session_state:
        st.session_state.input = ""
    
    # 界面头部
    st.markdown("""
    <div class="terminal-header">
        <h2>📈 Stock Analysis Terminal Pro</h2>
        <p>Professional Real-time Market Analysis • Stock Screening • Portfolio Comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 状态栏
    st.markdown("""
    <div class="status-bar">
        <div>🟢 SYSTEM ONLINE</div>
        <div>📡 REAL-TIME DATA</div>
        <div>🤖 AI ANALYSIS</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示消息历史
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "trader":
            st.markdown(f'<div class="user-message"><b>TRADER:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-message"><b>TERMINAL:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            if "chart" in msg:
                # 为每个图表生成唯一ID
                chart_id = hashlib.md5(f"{msg['content']}_{i}".encode()).hexdigest()
                st.plotly_chart(msg["chart"], use_container_width=True, key=f"chart_{chart_id}")
            if "results" in msg:
                st.dataframe(pd.DataFrame(msg["results"]))
    
    # 输入区域 - 添加回车键支持
    with st.form(key='command_form'):
        user_input = st.text_input(
            "Command (e.g., AAPL, screen PE<20, compare AAPL MSFT, TECH*)", 
            key="input",
            help="Press Enter to execute command"
        )
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("🚀 Execute")
        with col2:
            if st.form_submit_button("🗑️ Clear"):
                st.session_state.messages = []
                st.session_state.input = ""
                st.experimental_rerun()
    
    # 处理表单提交 - 避免重复处理相同命令
    if (submit_button or user_input) and user_input.strip():
        # 检查是否重复命令
        last_command = st.session_state.get("last_command", "")
        if user_input != last_command:
            st.session_state.last_command = user_input
            # 添加用户消息
            st.session_state.messages.append({"role": "trader", "content": user_input})
            process_command(user_input)
        else:
            st.session_state.last_command = ""

def process_command(command: str):
    """Process user command and display results"""
    analyzer = st.session_state.analyzer
    container = st.empty()
    
    for response in analyzer.process_command(command):
        if response["type"] == "status":
            container.markdown(f'<div class="ai-message">{response["content"]}</div>', unsafe_allow_html=True)
        elif response["type"] == "analysis":
            data = response["content"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Analysis for {data['symbol']}: Price ${data['price']:.2f}, Change {data['change']:.2f}%, PE {data['pe']}, RSI {data['rsi']:.1f}",
                "chart": data["chart"]
            })
            # 清除输入并刷新
            st.session_state.input = ""
            st.experimental_rerun()
        elif response["type"] == "screening":
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Found {len(results)} stocks matching criteria",
                "results": results
            })
            # 清除输入并刷新
            st.session_state.input = ""
            st.experimental_rerun()
        elif response["type"] == "comparison":
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Comparison of {len(results)} stocks",
                "results": results
            })
            # 清除输入并刷新
            st.session_state.input = ""
            st.experimental_rerun()
        elif response["type"] == "check_all":
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Found {len(results)} stocks starting with {response['content']['prefix']}",
                "results": results
            })
            # 清除输入并刷新
            st.session_state.input = ""
            st.experimental_rerun()
        elif response["type"] == "multiple_analysis":
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Analysis of {len(results)} stocks",
                "results": results
            })
            # 清除输入并刷新
            st.session_state.input = ""
            st.experimental_rerun()
        elif response["type"] == "error":
            st.session_state.messages.append({"role": "assistant", "content": response["content"]})
            # 清除输入并刷新
            st.session_state.input = ""
            st.experimental_rerun()
    
    # 在命令处理结束后确保清除输入
    st.session_state.input = ""
    st.experimental_rerun()

if __name__ == "__main__":
    main()
