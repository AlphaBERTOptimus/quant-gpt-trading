import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import re
import time
from typing import Dict, List, Generator, Any

# 精简版主题配置
def get_theme_css():
    return """
<style>
    :root {
        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --text-primary: #F8FAFC;
        --accent-color: #3B82F6;
        --success-color: #10B981;
    }
    .stApp { background: var(--bg-primary); color: var(--text-primary); }
    .terminal-header { background: linear-gradient(135deg, var(--bg-secondary) 0%, #334155 100%); padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; }
    .status-bar { display: flex; gap: 1rem; background: var(--bg-secondary); padding: 0.5rem 1rem; border-radius: 6px; margin-bottom: 1rem; font-size: 0.8rem; }
    .user-message { background: var(--bg-secondary); padding: 1rem; margin: 0.5rem 0; border-radius: 6px; border-left: 3px solid var(--accent-color); }
    .ai-message { background: var(--bg-secondary); padding: 1rem; margin: 0.5rem 0; border-radius: 6px; border-left: 3px solid var(--success-color); }
    .stButton>button { background: var(--accent-color); color: white; border-radius: 6px; }
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
        
        # 添加缺失的 analyze_multiple_stocks 方法
    def analyze_multiple_stocks(self, symbols: List[str]) -> Generator[Dict, None, None]:
        """Analyze multiple stocks"""
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
def process_command(command: str):
    analyzer = st.session_state.analyzer
    container = st.empty()
    
    for response in analyzer.process_command(command):
        if response["type"] == "status":
            container.markdown(f'<div class="ai-message">{response["content"]}</div>', unsafe_allow_html=True)
        elif response["type"] == "analysis":
            # 处理单个股票分析
            data = response["content"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Analysis for {data['symbol']}: Price ${data['price']:.2f}, Change {data['change']:.2f}%, PE {data['pe']}, RSI {data['rsi']:.1f}",
                "chart": data["chart"]
            })
            st.rerun()
        elif response["type"] == "screening":
            # 处理筛选结果
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Found {len(results)} stocks matching criteria",
                "results": results
            })
            st.rerun()
        elif response["type"] == "comparison":
            # 处理比较结果
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Comparison of {len(results)} stocks",
                "results": results
            })
            st.rerun()
        elif response["type"] == "check_all":
            # 处理前缀搜索
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Found {len(results)} stocks starting with {response['content']['prefix']}",
                "results": results
            })
            st.rerun()
        elif response["type"] == "multiple_analysis":  # 添加对新类型的处理
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Analysis of {len(results)} stocks",
                "results": results
            })
            st.rerun()
        elif response["type"] == "error":
            st.session_state.messages.append({"role": "assistant", "content": response["content"]})
            st.rerun()

    def analyze_stock(self, symbol: str) -> Generator[Dict, None, None]:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            info = ticker.info
            
            if data.empty:
                yield {"type": "error", "content": f"No data for {symbol}"}
                return
            
            # 核心分析步骤
            price = data['Close'].iloc[-1]
            change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
            pe = info.get('trailingPE', 'N/A')
            market_cap = f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A'
            
            # 精简技术指标
            rsi = self.calculate_rsi(data) if len(data) >= 14 else 'N/A'
            
            # 生成图表
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
            yield {"type": "error", "content": f"Error: {str(e)}"}

    def calculate_rsi(self, data: pd.DataFrame) -> float:
        close = data['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).iloc[-1]

    def create_chart(self, data: pd.DataFrame) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        # 价格图表
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ), row=1, col=1
        )
        
        # 成交量图表
        colors = ['green' if close > open else 'red' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Stock Analysis",
            height=600,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        
        return fig

    def screen_stocks(self, params: Dict) -> Generator[Dict, None, None]:
        stocks = self.stock_db.get_all_us_stocks()[:20]  # 限制数量
        results = []
        
        for symbol in stocks:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")
                info = ticker.info
                
                if not data.empty:
                    pe = info.get('trailingPE', 1000)
                    
                    # 检查筛选条件
                    if 'pe_max' in params and pe > params['pe_max']:
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
            yield {
                "type": "screening",
                "content": {
                    'results': results,
                    'criteria': params
                }
            }
        else:
            yield {"type": "error", "content": "No stocks found"}

    def compare_stocks(self, symbols: List[str]) -> Generator[Dict, None, None]:
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
            yield {"type": "error", "content": "Comparison failed"}

    def check_all_stocks(self, params: Dict) -> Generator[Dict, None, None]:
        prefix = params.get('prefix', '')
        all_stocks = self.stock_db.get_all_us_stocks()
        matching = [s for s in all_stocks if s.startswith(prefix)][:10]  # 限制数量
        results = []
        
        for symbol in matching:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")
                if not data.empty:
                    results.append({
                        'symbol': symbol,
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
            yield {"type": "error", "content": "No stocks found"}

# Streamlit界面
def main():
    st.set_page_config(page_title="Stock Terminal", layout="wide")
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
    # 初始化session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    
    # 界面头部
    st.markdown("""
    <div class="terminal-header">
        <h2>📈 Stock Analysis Terminal</h2>
        <p>Real-time market analysis and screening</p>
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
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><b>USER:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-message"><b>TERMINAL:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            if "chart" in msg:
                st.plotly_chart(msg["chart"], use_container_width=True)
            if "results" in msg:
                st.dataframe(pd.DataFrame(msg["results"]))
    
    # 输入区域
    user_input = st.text_input("Command (e.g., AAPL, screen PE<20, compare AAPL MSFT, TECH*)", key="input")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Execute", type="primary"):
            if user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input})
                process_command(user_input)
    with col2:
        if st.button("Clear"):
            st.session_state.messages = []
            st.rerun()

def process_command(command: str):
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
            st.rerun()
        elif response["type"] == "screening":
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Found {len(results)} stocks matching criteria",
                "results": results
            })
            st.rerun()
        elif response["type"] == "comparison":
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Comparison of {len(results)} stocks",
                "results": results
            })
            st.rerun()
        elif response["type"] == "check_all":
            results = response["content"]["results"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Found {len(results)} stocks starting with {response['content']['prefix']}",
                "results": results
            })
            st.rerun()
        elif response["type"] == "error":
            st.session_state.messages.append({"role": "assistant", "content": response["content"]})
            st.rerun()

if __name__ == "__main__":
    main()
