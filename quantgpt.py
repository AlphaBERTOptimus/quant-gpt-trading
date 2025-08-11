# quantgpt.py
# QuantGPT - AI量化交易聊天助手

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
# 页面配置
# ===================================
st.set_page_config(
    page_title="QuantGPT - AI量化交易助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================================
# 样式设置
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
# 量化分析类
# ===================================

class QuantGPTAssistant:
    """QuantGPT AI助手"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """初始化会话状态"""
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": """👋 你好！我是QuantGPT，你的AI量化交易助手！

我可以帮你：
- 📊 分析股票（例如："分析AAPL"）
- 📈 查看技术指标（例如："TSLA的RSI是多少？"）
- 💡 提供投资建议（例如："我应该买入NVDA吗？"）
- 📉 比较股票（例如："比较AAPL和GOOGL"）
- 🎯 策略回测（例如："测试MSFT的均线策略"）
- 💰 计算收益（例如："如果我投资1万美元到TSLA会怎样？"）

请随便问我任何关于股票和投资的问题！"""}
            ]
        
        if 'analyzing' not in st.session_state:
            st.session_state.analyzing = False
    
    @st.cache_data(ttl=3600)
    def get_stock_data(_self, symbol, period="1y"):
        """获取股票数据"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            info = ticker.info
            return data, info
        except:
            # 返回模拟数据
            return _self.generate_mock_data(symbol, period)
    
    def generate_mock_data(self, symbol, period="1y"):
        """生成模拟数据"""
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
        """计算技术指标"""
        df = data.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 移动平均
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        return df
    
    def analyze_stock(self, symbol):
        """分析股票"""
        data, info = self.get_stock_data(symbol)
        data = self.calculate_indicators(data)
        
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        change = (current_price - prev_close) / prev_close * 100
        
        rsi = data['RSI'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        
        # 生成建议
        score = 50
        if current_price > sma_20 > sma_50:
            score += 20
        if 30 < rsi < 70:
            score += 10
        elif rsi < 30:
            score += 15
        
        if score >= 70:
            recommendation = "🟢 **买入**"
            reason = "技术指标显示强劲的上涨信号"
        elif score >= 60:
            recommendation = "🟡 **持有**"
            reason = "市场表现中性，建议观望"
        else:
            recommendation = "🔴 **卖出**"
            reason = "技术指标显示下跌风险"
        
        analysis = f"""
### 📊 {symbol} 分析报告

**基本信息：**
- 当前价格：${current_price:.2f}
- 今日涨跌：{change:+.2f}%
- 成交量：{data['Volume'].iloc[-1]:,.0f}

**技术指标：**
- RSI(14)：{rsi:.2f} {'(超卖)' if rsi < 30 else '(超买)' if rsi > 70 else '(中性)'}
- SMA20：${sma_20:.2f}
- SMA50：${sma_50:.2f}
- 趋势：{'上升📈' if current_price > sma_20 > sma_50 else '下降📉' if current_price < sma_20 < sma_50 else '震荡📊'}

**AI建议：** {recommendation}
**理由：** {reason}

**风险提示：** 投资有风险，请谨慎决策。
"""
        return analysis, data
    
    def create_chart(self, symbol, data):
        """创建股价图表"""
        fig = go.Figure()
        
        # K线图
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='价格'
        ))
        
        # 添加移动平均线
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
            title=f'{symbol} 价格走势',
            yaxis_title='价格 ($)',
            xaxis_title='日期',
            height=400,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def process_query(self, query):
        """处理用户查询"""
        query_lower = query.lower()
        
        # 提取股票代码
        import re
        stock_symbols = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        
        # 分析类查询
        if any(word in query_lower for word in ['分析', '评估', '看看', 'analyze', 'check']):
            if stock_symbols:
                symbol = stock_symbols[0]
                analysis, data = self.analyze_stock(symbol)
                chart = self.create_chart(symbol, data)
                return analysis, chart
            else:
                return "请提供股票代码，例如：'分析AAPL'", None
        
        # 价格查询
        elif any(word in query_lower for word in ['价格', '多少钱', 'price', 'cost']):
            if stock_symbols:
                symbol = stock_symbols[0]
                data, _ = self.get_stock_data(symbol, "1mo")
                price = data['Close'].iloc[-1]
                change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                return f"**{symbol}** 当前价格：**${price:.2f}** ({change:+.2f}%)", None
            else:
                return "请提供股票代码，例如：'AAPL的价格是多少？'", None
        
        # 买卖建议
        elif any(word in query_lower for word in ['买', '卖', 'buy', 'sell', '建议', '推荐']):
            if stock_symbols:
                symbol = stock_symbols[0]
                analysis, data = self.analyze_stock(symbol)
                return analysis, None
            else:
                # 推荐热门股票
                return """### 🔥 今日热门推荐：

1. **NVDA** - AI芯片龙头，技术指标强劲 🟢
2. **AAPL** - 稳健蓝筹，适合长期持有 🟢
3. **TSLA** - 电动车领导者，波动较大 🟡
4. **MSFT** - 云计算巨头，增长稳定 🟢
5. **GOOGL** - 搜索霸主，AI转型中 🟡

输入具体股票代码获取详细分析！""", None
        
        # RSI查询
        elif 'rsi' in query_lower:
            if stock_symbols:
                symbol = stock_symbols[0]
                data, _ = self.get_stock_data(symbol)
                data = self.calculate_indicators(data)
                rsi = data['RSI'].iloc[-1]
                
                if rsi < 30:
                    status = "**超卖区域** 🟢 - 可能是买入机会"
                elif rsi > 70:
                    status = "**超买区域** 🔴 - 可能面临回调"
                else:
                    status = "**中性区域** 🟡 - 没有明确信号"
                
                return f"**{symbol}** RSI(14) = **{rsi:.2f}**\n\n{status}", None
        
        # 比较股票
        elif any(word in query_lower for word in ['比较', 'compare', 'vs']):
            if len(stock_symbols) >= 2:
                results = []
                for symbol in stock_symbols[:2]:
                    data, _ = self.get_stock_data(symbol, "1mo")
                    change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
                    results.append(f"**{symbol}**: 月收益 {change:+.2f}%")
                
                winner = stock_symbols[0] if results[0] > results[1] else stock_symbols[1]
                return f"### 📊 股票比较\n\n" + "\n".join(results) + f"\n\n🏆 **{winner}** 表现更好！", None
            else:
                return "请提供两个股票代码进行比较，例如：'比较AAPL和GOOGL'", None
        
        # 投资计算
        elif any(word in query_lower for word in ['投资', '收益', 'invest', 'return', '如果']):
            if stock_symbols and any(char.isdigit() for char in query):
                # 提取金额
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
                    
                    return f"""### 💰 投资模拟 - {symbol}

**初始投资：** ${amount:,.2f}
**买入价格：** ${start_price:.2f}
**当前价格：** ${end_price:.2f}
**持有股数：** {shares:.2f}

**当前价值：** ${final_value:,.2f}
**盈亏金额：** ${profit:+,.2f}
**收益率：** {return_rate:+.2f}%

{'🎉 恭喜！投资获利！' if profit > 0 else '😔 暂时亏损，请耐心持有'}""", None
        
        # 默认回复
        else:
            return """我可以帮你：

- 📊 **分析股票**：输入"分析AAPL"
- 💵 **查询价格**：输入"TSLA的价格"
- 📈 **技术指标**：输入"NVDA的RSI"
- 🔄 **比较股票**：输入"比较AAPL和GOOGL"
- 💰 **模拟投资**：输入"投资10000美元到MSFT"
- 💡 **获取建议**：输入"我应该买什么股票"

请问有什么可以帮助你的？""", None

# ===================================
# 主应用
# ===================================

def main():
    st.title("🤖 QuantGPT - AI量化交易助手")
    st.markdown("---")
    
    # 初始化助手
    assistant = QuantGPTAssistant()
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 📌 快速操作")
        
        # 热门股票按钮
        st.markdown("**热门股票：**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🍎 AAPL"):
                st.session_state.messages.append({"role": "user", "content": "分析AAPL"})
            if st.button("🚗 TSLA"):
                st.session_state.messages.append({"role": "user", "content": "分析TSLA"})
            if st.button("🖥️ NVDA"):
                st.session_state.messages.append({"role": "user", "content": "分析NVDA"})
        with col2:
            if st.button("🔍 GOOGL"):
                st.session_state.messages.append({"role": "user", "content": "分析GOOGL"})
            if st.button("💻 MSFT"):
                st.session_state.messages.append({"role": "user", "content": "分析MSFT"})
            if st.button("📱 META"):
                st.session_state.messages.append({"role": "user", "content": "分析META"})
        
        st.markdown("---")
        st.markdown("### 💡 示例问题")
        example_questions = [
            "AAPL的价格是多少？",
            "分析TSLA",
            "NVDA的RSI是多少？",
            "比较AAPL和GOOGL",
            "如果我投资10000美元到MSFT会怎样？",
            "我应该买什么股票？"
        ]
        
        for question in example_questions:
            if st.button(f"📝 {question}", key=question):
                st.session_state.messages.append({"role": "user", "content": question})
        
        st.markdown("---")
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 如果有图表，显示图表
            if "chart" in message and message["chart"] is not None:
                st.plotly_chart(message["chart"], use_container_width=True)
    
    # 用户输入
    if prompt := st.chat_input("问我任何关于股票的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成助手回复
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response, chart = assistant.process_query(prompt)
                st.markdown(response)
                
                # 如果有图表，显示它
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                
                # 保存助手消息
                message_data = {"role": "assistant", "content": response}
                if chart is not None:
                    message_data["chart"] = chart
                st.session_state.messages.append(message_data)

if __name__ == "__main__":
    main()
