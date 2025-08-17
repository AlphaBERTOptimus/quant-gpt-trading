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
    .stApp { 
        background: var(--bg-primary); 
        color: var(--text-primary); 
        font-family: 'SF Mono', 'Courier New', monospace;
    }
    .terminal-header { 
        background: linear-gradient(135deg, var(--bg-secondary) 0%, #334155 100%); 
        padding: 1.5rem; 
        border-radius: 8px; 
        margin-bottom: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .status-bar { 
        display: flex; 
        gap: 1.2rem; 
        background: var(--bg-secondary); 
        padding: 0.8rem 1.2rem; 
        border-radius: 8px; 
        margin-bottom: 1.5rem; 
        font-size: 0.85rem;
        border: 1px solid #334155;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        justify-content: space-between;
    }
    .user-message { 
        background: var(--bg-secondary); 
        padding: 1.2rem; 
        margin: 0.8rem 0; 
        border-radius: 8px; 
        border-left: 4px solid var(--accent-color);
        font-size: 1.0rem;
    }
    .ai-message { 
        background: var(--bg-secondary); 
        padding: 1.2rem; 
        margin: 0.8rem 0; 
        border-radius: 8px; 
        border-left: 4px solid var(--success-color);
        font-size: 1.0rem;
    }
    .error-message {
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid var(--danger-color) !important;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    .status-message {
        background: rgba(59, 130, 246, 0.15) !important;
        border-left: 4px solid var(--accent-color) !important;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    
    /* 修复按钮样式 */
    .stButton>button {
        background: var(--accent-color) !important;
        color: white !important;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.8rem 1.1rem !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: var(--success-color) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* 修复输入框 */
    .stTextInput>div>div>input {
        color: var(--text-primary) !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid #334155 !important;
        padding: 0.9rem !important;
        border-radius: 6px;
        font-size: 1.05rem;
    }
    .stTextInput>div>div>input:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    .stTextInput>div>div>input::placeholder {
        color: #94a3b8 !important;
    }
    
    /* 表格样式 */
    .stDataFrame {
        border: 1px solid #334155 !important;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    .stDataFrame th {
        background-color: var(--bg-secondary) !important;
    }
    
    @media (max-width: 768px) {
        .status-bar {
            flex-direction: column;
            gap: 0.5rem;
        }
        .stButton>button {
            padding: 0.75rem !important;
            font-size: 0.9rem !important;
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
                    pe_match = re.search(r'PE\s*<\s*(\d+)', text)
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
        """处理用户命令的路由"""
        parsed = self.parser.parse_command(text)
        yield {"type": "status", "content": f"🔍 Processing command: {text}"}
        time.sleep(0.3)
        
        try:
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
        except Exception as e:
            yield {"type": "error", "content": f"🔥 Command processing failed: {str(e)}"}
    
    def get_stock_data(self, symbol: str, retries=2):
        """获取股票数据，带有重试机制"""
        for i in range(retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                info = ticker.info
                
                if not data.empty:
                    return data, info
                time.sleep(0.5)
            except Exception:
                time.sleep(1)
        return None, None
    
    def analyze_stock(self, symbol: str) -> Generator[Dict, None, None]:
        """分析单个股票"""
        yield {"type": "status", "content": f"📊 Analyzing {symbol}..."}
        time.sleep(0.5)
        
        try:
            data, info = self.get_stock_data(symbol)
            
            if data is None or data.empty:
                yield {"type": "error", "content": f"⚠️ No data found for {symbol}"}
                return
            
            # 计算关键指标
            price = data['Close'].iloc[-1]
            change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
            pe = info.get('trailingPE', 'N/A')
            market_cap = f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A'
            
            # 计算RSI
            rsi = self.calculate_rsi(data) if len(data) >= 14 else 'N/A'
            
            # 创建图表
            fig = self.create_chart(data, symbol)
            
            # 生成分析报告
            analysis_report = self.generate_analysis_report(symbol, info)
            
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
            yield {"type": "error", "content": f"❌ Analysis failed for {symbol}: {str(e)}"}
    
    def generate_analysis_report(self, symbol: str, info: dict) -> str:
        """生成专业的股票分析报告"""
        try:
            name = info.get('longName', symbol)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            recommendation = info.get('recommendationKey', 'N/A').capitalize()
            
            report = f"""
            <div class='ai-message'>
                <h3>📈 {name} ({symbol}) Professional Analysis</h3>
                <p><b>Sector:</b> {sector} | <b>Industry:</b> {industry}</p>
                <p><b>Valuation:</b> PE Ratio: {info.get('trailingPE', 'N/A')}</p>
                <p><b>Market Cap:</b> ${info.get('marketCap', 0)/1e9:.2f}B</p>
                
                <h4>Key Metrics:</h4>
                <ul>
                    <li><b>Profit Margin:</b> {info.get('profitMargins', 'N/A')}</li>
                    <li><b>Revenue Growth (YoY):</b> {info.get('revenueGrowth', 'N/A')}</li>
                    <li><b>52 Week Range:</b> {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}</li>
                </ul>
                
                <h4>Analyst Consensus:</h4>
                <p>Recommendation: <b>{recommendation}</b> | Target Price: ${info.get('targetMeanPrice', 'N/A')}</p>
            </div>
            """
            return report
        except:
            return f"<div class='ai-message'>📋 Generated basic analysis report for {symbol}</div>"
    
    def analyze_multiple_stocks(self, symbols: List[str]) -> Generator[Dict, None, None]:
        """分析多只股票"""
        yield {"type": "status", "content": f"📊 Analyzing {len(symbols)} stocks: {', '.join(symbols)}"}
        time.sleep(0.5)
        
        results = []
        for i, symbol in enumerate(symbols):
            yield {"type": "status", "content": f"🔄 Analyzing {symbol} ({i+1}/{len(symbols)})..."}
            time.sleep(0.3)
            
            try:
                data, info = self.get_stock_data(symbol)
                
                if data is not None and not data.empty:
                    price = data['Close'].iloc[-1]
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
                    pe = info.get('trailingPE', 'N/A')
                    market_cap = f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A'
                    
                    results.append({
                        'Symbol': symbol,
                        'Name': info.get('shortName', symbol)[:20],
                        'Price': price,
                        'Change (%)': f"{change:.2f}%",
                        'PE Ratio': pe,
                        'Market Cap': market_cap
                    })
            except:
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
        """计算RSI指标"""
        close = data['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).iloc[-1]
    
    def create_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """创建股票价格图表"""
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )
        
        # 价格图表
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
        
        # 添加50日均线
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
        
        # 成交量图表
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
        """根据条件筛选股票"""
        yield {"type": "status", "content": "🔍 Screening stocks based on your criteria..."}
        time.sleep(0.5)
        
        # 获取股票池
        stocks = self.stock_db.get_all_us_stocks()[:20]  # 演示用限制
        
        yield {"type": "status", "content": f"📊 Analyzing {len(stocks)} stocks..."}
        time.sleep(0.3)
        
        results = []
        for i, symbol in enumerate(stocks):
            try:
                _, info = self.get_stock_data(symbol)
                
                if info:
                    pe = info.get('trailingPE', 1000)
                    
                    # 检查条件
                    if 'pe_max' in criteria and pe > criteria['pe_max']:
                        continue
                    
                    # 获取当前价格
                    ticker = yf.Ticker(symbol)
                    current_data = ticker.history(period="1d")
                    price = current_data['Close'].iloc[-1] if not current_data.empty else 'N/A'
                        
                    results.append({
                        'Symbol': symbol,
                        'Price': price,
                        'PE Ratio': pe,
                        'Market Cap': f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A'
                    })
            except:
                continue
        
        if results:
            # 按PE排序
            results.sort(key=lambda x: x.get('PE Ratio', 999))
            
            yield {
                "type": "screening",
                "content": {
                    'results': results[:10],  # 显示前10个结果
                    'criteria': criteria
                }
            }
        else:
            yield {"type": "error", "content": "❌ No stocks found matching your criteria"}
    
    def compare_stocks(self, symbols: List[str]) -> Generator[Dict, None, None]:
        """比较多只股票"""
        yield {"type": "status", "content": f"⚖️ Comparing {len(symbols)} stocks: {', '.join(symbols)}"}
        time.sleep(0.5)
        
        results = []
        for symbol in symbols:
            try:
                data, info = self.get_stock_data(symbol)
                
                if data is not None and not data.empty:
                    price = data['Close'].iloc[-1]
                    ytd_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100) if len(data) > 0 else 0
                    results.append({
                        'Symbol': symbol,
                        'Current Price': price,
                        'YTD Change (%)': f"{ytd_change:.2f}%",
                        'PE Ratio': info.get('trailingPE', 'N/A')
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
        """查找匹配模式的股票"""
        prefix = params.get('prefix', '')
        yield {"type": "status", "content": f"🔍 Finding all stocks starting with '{prefix}'..."}
        time.sleep(0.5)
        
        # 获取股票池
        all_stocks = self.stock_db.get_all_us_stocks()
        matching = [s for s in all_stocks if s.startswith(prefix)][:10]  # 限制10个
        
        if not matching:
            yield {"type": "error", "content": f"❌ No stocks found starting with '{prefix}'"}
            return
        
        yield {"type": "status", "content": f"📊 Found {len(matching)} matching stocks. Analyzing..."}
        time.sleep(0.3)
        
        results = []
        for symbol in matching:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                info = ticker.info
                
                if not data.empty:
                    results.append({
                        'Symbol': symbol,
                        'Name': info.get('shortName', symbol)[:25],
                        'Price': data['Close'].iloc[-1],
                        'Volume': f"{data['Volume'].iloc[-1]/1e6:.1f}M"
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

# 主要Streamlit界面
def main():
    # 初始化会话状态 - 解决状态管理冲突
    if 'terminal_state' not in st.session_state:
        st.session_state.terminal_state = {
            'messages': [],
            'last_input': "",
            'processing': False
        }
    
    # 设置页面配置
    st.set_page_config(
        page_title="Stock Terminal Pro", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
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
        <div>🚀 v1.2.1</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示消息历史
    for i, msg in enumerate(st.session_state.terminal_state['messages']):
        if msg["role"] == "trader":
            st.markdown(f'<div class="user-message"><b>TRADER:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # 根据消息类型设置样式
            if msg.get("is_error", False):
                st.markdown(f'<div class="error-message"><b>TERMINAL:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            elif msg.get("is_status", False):
                st.markdown(f'<div class="status-message"><b>SYSTEM:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message"><b>TERMINAL:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            
            # 显示图表
            if "chart" in msg:
                chart_id = hashlib.md5(f"{msg['content']}_{i}".encode()).hexdigest()
                st.plotly_chart(msg["chart"], use_container_width=True, key=f"chart_{chart_id}")
            
            # 显示分析报告
            if "analysis_report" in msg:
                st.markdown(msg["analysis_report"], unsafe_allow_html=True)
            
            # 显示表格结果
            if "results" in msg:
                df = pd.DataFrame(msg["results"])
                if not df.empty:
                    # 格式化数字列
                    if 'Price' in df.columns:
                        df['Price'] = df['Price'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
                    st.dataframe(df)
    
    # 输入区域 - 解决状态管理问题
    with st.form(key='command_form'):
        user_input = st.text_input(
            "📋 Enter command (e.g., ANALYZE NVDA, screen PE<20, compare AAPL MSFT, TECH*)", 
            key="command_input",
            placeholder="Enter stock symbol or command...",
            value=st.session_state.terminal_state.get('last_input', ''),
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("🚀 EXECUTE")
        with col2:
            clear_button = st.form_submit_button("🗑️ CLEAR")
    
    # 表单提交处理
    if clear_button:
        st.session_state.terminal_state['messages'] = []
        st.session_state.terminal_state['last_input'] = ""
        st.session_state.terminal_state['processing'] = False
        st.experimental_rerun()
    
    if submit_button and user_input.strip() and not st.session_state.terminal_state.get('processing', False):
        # 更新状态
        st.session_state.terminal_state['last_input'] = user_input
        st.session_state.terminal_state['processing'] = True
        
        # 添加用户消息
        st.session_state.terminal_state['messages'].append({
            "role": "trader", 
            "content": user_input
        })
        
        # 初始化分析器
        analyzer = StockAnalyzer()
        
        # 处理命令
        for response in analyzer.process_command(user_input):
            if response["type"] == "status":
                st.session_state.terminal_state['messages'].append({
                    "role": "assistant",
                    "is_status": True,
                    "content": response["content"]
                })
                st.experimental_rerun()
            elif response["type"] in ["analysis", "screening", "comparison", "check_all", "multiple_analysis"]:
                # 确保所有响应中都有安全的内容
                safe_response = {
                    "role": "assistant",
                    "content": f"{response['type']} completed"
                }
                
                # 安全添加所有键值对
                for key in response.get("content", {}):
                    safe_response[key] = response["content"][key]
                
                # 确保分析报告键存在
                if response["type"] == "analysis" and "analysis_report" not in safe_response:
                    safe_response["analysis_report"] = "<div class='ai-message'>Basic analysis report generated</div>"
                
                st.session_state.terminal_state['messages'].append(safe_response)
                st.session_state.terminal_state['processing'] = False
                st.experimental_rerun()
            elif response["type"] == "error":
                st.session_state.terminal_state['messages'].append({
                    "role": "assistant",
                    "is_error": True,
                    "content": response["content"]
                })
                st.session_state.terminal_state['processing'] = False
                st.experimental_rerun()
        
        # 处理完成后重置状态
        st.session_state.terminal_state['processing'] = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()
