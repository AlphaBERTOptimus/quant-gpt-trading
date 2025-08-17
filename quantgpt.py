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

# 优化后的主题CSS - 修复所有可见性问题
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
        --button-text: #FFFFFF;
    }
    .stApp { 
        background: var(--bg-primary); 
        color: var(--text-primary); 
        font-family: 'Courier New', monospace;
    }
    .terminal-header { 
        background: linear-gradient(135deg, var(--bg-secondary) 0%, #334155 100%); 
        padding: 1.5rem; 
        border-radius: 8px; 
        margin-bottom: 1rem; 
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-bar { 
        display: flex; 
        gap: 1rem; 
        background: var(--bg-secondary); 
        padding: 0.8rem 1.2rem; 
        border-radius: 6px; 
        margin-bottom: 1.5rem; 
        font-size: 0.85rem;
        border: 1px solid #334155;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .user-message { 
        background: var(--bg-secondary); 
        padding: 1.2rem; 
        margin: 0.7rem 0; 
        border-radius: 8px; 
        border-left: 4px solid var(--accent-color);
        font-size: 0.95rem;
    }
    .ai-message { 
        background: var(--bg-secondary); 
        padding: 1.2rem; 
        margin: 0.7rem 0; 
        border-radius: 8px; 
        border-left: 4px solid var(--success-color);
        font-size: 0.95rem;
    }
    
    /* 修复按钮样式 - 确保高可见性 */
    .stButton>button {
        background: var(--accent-color) !important;
        color: var(--button-text) !important;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        padding: 0.7rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: var(--success-color) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* 修复输入框文本颜色 */
    .stTextInput>div>div>input {
        color: var(--text-primary) !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid #334155;
        padding: 0.8rem;
        border-radius: 6px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }
    
    /* 输入框标签 */
    .stTextInput label {
        color: #94A3B8 !important;
        font-weight: 500;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }
    
    /* 错误消息样式 */
    .error-message {
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid var(--danger-color) !important;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.7rem 0;
    }
    
    /* 状态消息样式 */
    .status-message {
        background: rgba(59, 130, 246, 0.15) !important;
        border-left: 4px solid var(--accent-color) !important;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.7rem 0;
    }
    
    /* 响应式布局 */
    @media (max-width: 768px) {
        .status-bar {
            flex-direction: column;
            gap: 0.5rem;
        }
        .stButton>button {
            padding: 0.6rem;
            font-size: 0.9rem;
        }
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
        self.cache = {}
    
    def process_command(self, text: str) -> Generator[Dict, None, None]:
        """Process user command and route to appropriate analysis"""
        parsed = self.parser.parse_command(text)
        yield {"type": "status", "content": f"🔍 Processing command: {text}"}
        time.sleep(0.5)  # 更自然的信息流
        
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
    
    def get_ticker_data(self, symbol: str, cache_key: str, retries=2):
        """获取ticker数据，带缓存和重试机制"""
        cache_key = f"{symbol}_{cache_key}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            info = ticker.info
            
            if data.empty and retries > 0:
                time.sleep(1)  # 稍等后重试
                return self.get_ticker_data(symbol, cache_key, retries-1)
                
            self.cache[cache_key] = (ticker, data, info)
            return ticker, data, info
            
        except Exception as e:
            if retries > 0:
                time.sleep(1)
                return self.get_ticker_data(symbol, cache_key, retries-1)
            raise e
    
    def analyze_stock(self, symbol: str) -> Generator[Dict, None, None]:
        """Analyze a single stock"""
        yield {"type": "status", "content": f"📊 Analyzing {symbol}..."}
        time.sleep(0.3)
        
        try:
            _, data, info = self.get_ticker_data(symbol, "analysis")
            
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
            fig = self.create_chart(data, symbol)
            
            # 生成专业分析报告
            analysis_report = self.generate_analysis_report(symbol, price, change, pe, rsi, info)
            
            yield {
                "type": "analysis",
                "content": {
                    'symbol': symbol,
                    'price': price,
                    'change': change,
                    'pe': pe,
                    'market_cap': market_cap,
                    'rsi': rsi,
                    'chart': fig,
                    'analysis_report': analysis_report
                }
            }
            
        except Exception as e:
            yield {"type": "error", "content": f"❌ Failed to analyze {symbol}: {str(e)}"}
    
    def generate_analysis_report(self, symbol: str, price: float, change: float, 
                               pe: float, rsi: float, info: dict) -> str:
        """生成专业的股票分析报告"""
        name = info.get('longName', symbol)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        recommendation = info.get('recommendationKey', 'N/A').capitalize()
        
        report = f"""
        <div class='ai-message'>
            <h3>📈 {name} ({symbol}) Professional Analysis</h3>
            <p><b>Sector:</b> {sector} | <b>Industry:</b> {industry}</p>
            <p><b>Current Price:</b> ${price:.2f} | <b>Change:</b> {change:.2f}%</p>
            <p><b>Valuation:</b> PE Ratio: {pe} | RSI(14): {rsi if isinstance(rsi, str) else rsi:.1f}</p>
            
            <h4>Key Metrics:</h4>
            <ul>
                <li><b>Market Cap:</b> ${info.get('marketCap', 0)/1e9:.2f}B</li>
                <li><b>Profit Margin:</b> {info.get('profitMargins', 'N/A')}</li>
                <li><b>Revenue Growth (YoY):</b> {info.get('revenueGrowth', 'N/A')}</li>
            </ul>
            
            <h4>Analyst Consensus:</h4>
            <p>Recommendation: <b>{recommendation}</b></p>
            
            <h4>Trading Strategy:</h4>
            <p>Technical Indicators show {self.get_market_sentiment(rsi)} market sentiment.</p>
            <p>For swing trading, consider {self.get_trading_recommendation(rsi, change)}</p>
        </div>
        """
        return report
    
    def get_market_sentiment(self, rsi: float) -> str:
        """根据RSI确定市场情绪"""
        if isinstance(rsi, str):
            return "neutral"
        if rsi < 30:
            return "oversold (bullish)"
        elif rsi > 70:
            return "overbought (bearish)"
        return "neutral"
    
    def get_trading_recommendation(self, rsi: float, change: float) -> str:
        """生成交易建议"""
        if isinstance(rsi, str):
            return "monitoring price action"
            
        if rsi < 30 and change > 0:
            return "accumulating on dips"
        elif rsi > 70 and change < 0:
            return "taking profits"
        elif 40 < rsi < 60:
            return "holding positions"
        return "watching for breakout opportunities"
    
    def analyze_multiple_stocks(self, symbols: List[str]) -> Generator[Dict, None, None]:
        """Analyze multiple stocks"""
        yield {"type": "status", "content": f"📊 Analyzing {len(symbols)} stocks: {', '.join(symbols)}"}
        time.sleep(0.5)
        
        results = []
        for i, symbol in enumerate(symbols):
            yield {"type": "status", "content": f"🔄 Analyzing {symbol} ({i+1}/{len(symbols)})..."}
            time.sleep(0.2)
            
            try:
                _, data, info = self.get_ticker_data(symbol, "multi_analysis")
                
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
                    pe = info.get('trailingPE', 'N/A')
                    market_cap = f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A'
                    
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol)[:40],
                        'price': price,
                        'change': f"{change:.2f}%",
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
    
    def create_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create stock price chart"""
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )
        
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
        
        # Add 50-day moving average
        ma50 = data['Close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma50,
                name='MA 50',
                line=dict(color='#3B82F6', width=2)
            ),
            row=1, col=1
        )
        
        # Volume chart
        colors = ['#10B981' if close > open else '#EF4444' 
                 for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(
                x=data.index, 
                y=data['Volume'], 
                name='Volume', 
                marker_color=colors, 
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{symbol} Stock Analysis",
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode="x unified",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def screen_stocks(self, criteria: Dict) -> Generator[Dict, None, None]:
        """Screen stocks based on criteria"""
        yield {"type": "status", "content": "🔍 Screening stocks based on your criteria..."}
        time.sleep(0.5)
        
        # Get stock universe
        stocks = self.stock_db.get_all_us_stocks()[:20]  # Limit for demo
        
        yield {"type": "status", "content": f"📊 Analyzing {len(stocks)} stocks..."}
        time.sleep(0.3)
        
        results = []
        for i, symbol in enumerate(stocks):
            if (i + 1) % 5 == 0:
                yield {"type": "status", "content": f"🔄 Processed {i+1}/{len(stocks)} stocks..."}
                time.sleep(0.1)
            
            try:
                _, data, info = self.get_ticker_data(symbol, "screening")
                
                if not data.empty:
                    pe = info.get('trailingPE', 1000)
                    
                    # Check criteria
                    if 'pe_max' in criteria and pe > criteria['pe_max']:
                        continue
                        
                    results.append({
                        'symbol': symbol,
                        'price': data['Close'].iloc[-1],
                        'pe': pe,
                        'market_cap': f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A'
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
        time.sleep(0.5)
        
        results = []
        for symbol in symbols:
            try:
                _, data, info = self.get_ticker_data(symbol, "comparison")
                
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100)
                    results.append({
                        'symbol': symbol,
                        'price': price,
                        'ytd_change': f"{change:.2f}%",
                        'pe': info.get('trailingPE', 'N/A')
                    })
            except Exception as e:
                yield {"type": "status", "content": f"⚠️ Could not analyze {symbol}: {str(e)}"}
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
        time.sleep(0.5)
        
        # Get stock universe
        all_stocks = self.stock_db.get_all_us_stocks()
        matching = [s for s in all_stocks if s.startswith(prefix)][:10]  # Limit to 10
        
        if not matching:
            yield {"type": "error", "content": f"❌ No stocks found starting with '{prefix}'"}
            return
        
        yield {"type": "status", "content": f"📊 Found {len(matching)} matching stocks. Analyzing..."}
        time.sleep(0.3)
        
        results = []
        for symbol in matching:
            try:
                _, data, info = self.get_ticker_data(symbol, "check_all")
                
                if not data.empty:
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol)[:40],
                        'price': data['Close'].iloc[-1],
                        'volume': f"{data['Volume'].iloc[-1]/1e6:.1f}M"
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
    st.set_page_config(
        page_title="Stock Terminal Pro", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
    # 初始化session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    if "input" not in st.session_state:
        st.session_state.input = ""
    if "last_command" not in st.session_state:
        st.session_state.last_command = ""
    
    # 界面头部
    st.markdown("""
    <div class="terminal-header">
        <h1>📈 Stock Analysis Terminal Pro</h1>
        <p>Professional Real-time Market Analysis • Stock Screening • Portfolio Comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 状态栏
    st.markdown("""
    <div class="status-bar">
        <div>🟢 SYSTEM ONLINE</div>
        <div>📡 REAL-TIME DATA</div>
        <div>🤖 AI ANALYSIS</div>
        <div>🚀 v1.2.0</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示消息历史
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "trader":
            st.markdown(f'<div class="user-message"><b>TRADER:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # 根据消息类型使用不同的样式
            if msg.get("is_error", False):
                st.markdown(f'<div class="error-message"><b>TERMINAL:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            elif msg.get("is_status", False):
                st.markdown(f'<div class="status-message"><b>SYSTEM:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message"><b>TERMINAL:</b> {msg["content"]}</div>', unsafe_allow_html=True)
                
            if "chart" in msg:
                # 为每个图表生成唯一ID
                chart_id = hashlib.md5(f"{msg['content']}_{i}".encode()).hexdigest()
                st.plotly_chart(msg["chart"], use_container_width=True, key=f"chart_{chart_id}")
                
            if "results" in msg:
                df = pd.DataFrame(msg["results"])
                st.dataframe(df.style.format({
                    'price': '${:.2f}',
                    'change': '{}',
                    'ytd_change': '{}',
                    'pe': '{:.2f}' if df['pe'].dtype != 'object' else None
                }))
                
            if "analysis_report" in msg:
                st.markdown(msg["analysis_report"], unsafe_allow_html=True)
    
    # 输入区域
    with st.form(key='command_form'):
        col_input, col_execute = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "📋 Enter command (e.g., AAPL, screen PE<20, compare AAPL MSFT, TECH*)", 
                key="input",
                placeholder="Enter stock symbol or command...",
                label_visibility="collapsed"
            )
        with col_execute:
            st.write("")  # 用于垂直居中
            st.write("")
            execute_button = st.form_submit_button("🚀 RUN", use_container_width=True)
        
        clear_col1, clear_col2 = st.columns([1, 1])
        with clear_col1:
            pass
        with clear_col2:
            clear_button = st.form_submit_button("🗑️ CLEAR", use_container_width=True)
    
    # 处理表单提交
    if clear_button:
        st.session_state.messages = []
        st.session_state.input = ""
        st.session_state.last_command = ""
        st.experimental_rerun()
    
    if (execute_button or user_input) and user_input.strip():
        # 检查是否重复命令
        if user_input != st.session_state.last_command:
            st.session_state.last_command = user_input
            # 添加用户消息
            st.session_state.messages.append({"role": "trader", "content": user_input})
            process_command(user_input)

def process_command(command: str):
    """Process user command and display results"""
    analyzer = st.session_state.analyzer
    status_container = st.empty()
    
    try:
        for response in analyzer.process_command(command):
            if response["type"] == "status":
                status_container.markdown(
                    f'<div class="status-message">{response["content"]}</div>', 
                    unsafe_allow_html=True
                )
            elif response["type"] == "analysis":
                data = response["content"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ Analysis complete for {data['symbol']}",
                    "chart": data["chart"],
                    "analysis_report": data["analysis_report"]
                })
                # 清除输入并刷新
                st.session_state.input = ""
                st.experimental_rerun()
            elif response["type"] == "screening":
                results = response["content"]["results"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"🔍 Found {len(results)} stocks matching criteria",
                    "results": results
                })
                st.session_state.input = ""
                st.experimental_rerun()
            elif response["type"] == "comparison":
                results = response["content"]["results"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚖️ Comparison of {len(results)} stocks",
                    "results": results
                })
                st.session_state.input = ""
                st.experimental_rerun()
            elif response["type"] == "check_all":
                results = response["content"]["results"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"🔍 Found {len(results)} stocks starting with {response['content']['prefix']}",
                    "results": results
                })
                st.session_state.input = ""
                st.experimental_rerun()
            elif response["type"] == "multiple_analysis":
                results = response["content"]["results"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"📊 Analysis of {len(results)} stocks",
                    "results": results
                })
                st.session_state.input = ""
                st.experimental_rerun()
            elif response["type"] == "error":
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["content"],
                    "is_error": True
                })
                st.session_state.input = ""
                st.experimental_rerun()
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"🔥 Critical Error: {str(e)}",
            "is_error": True
        })
        st.session_state.input = ""
        st.experimental_rerun()
    
    # 在命令处理结束后确保清除输入
    st.session_state.input = ""
    st.experimental_rerun()

if __name__ == "__main__":
    main()
