# quantgpt.py
# QuantGPT Web应用 - 适用于Streamlit部署

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===================================
# 页面配置
# ===================================
st.set_page_config(
    page_title="QuantGPT - AI量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================
# 样式设置
# ===================================
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===================================
# 核心功能类（简化版）
# ===================================

class SimpleQuantAnalyzer:
    """简化的量化分析器"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_stock_data(symbol, period="1y"):
        """获取股票数据"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            info = ticker.info
            return data, info
        except Exception as e:
            st.error(f"获取{symbol}数据失败: {e}")
            return None, None
    
    @staticmethod
    def calculate_technical_indicators(data):
        """计算技术指标"""
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # 移动平均
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
    
    @staticmethod
    def generate_ai_recommendation(data, info):
        """生成AI建议（简化版）"""
        if data is None or data.empty:
            return "数据不足", "无法分析", 0
        
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data else current_price
        sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data else current_price
        rsi = data['RSI'].iloc[-1] if 'RSI' in data else 50
        
        score = 50
        
        # 技术面评分
        if current_price > sma_20 > sma_50:
            score += 20
        if 30 < rsi < 70:
            score += 10
        elif rsi < 30:
            score += 15
        
        # 基本面评分
        pe_ratio = info.get('trailingPE', 0) if info else 0
        if 0 < pe_ratio < 20:
            score += 15
        
        # 生成建议
        if score >= 70:
            recommendation = "🟢 买入"
            reason = "技术指标和基本面都显示积极信号"
        elif score >= 60:
            recommendation = "🟡 持有"
            reason = "整体表现中性，建议观望"
        else:
            recommendation = "🔴 卖出"
            reason = "多项指标显示负面信号"
        
        confidence = min(abs(score - 50) / 50, 1.0)
        
        return recommendation, reason, confidence

# ===================================
# 主应用界面
# ===================================

def main():
    # 侧边栏
    with st.sidebar:
        st.title("🚀 QuantGPT")
        st.markdown("### AI驱动的量化交易系统")
        
        # 功能选择
        page = st.selectbox(
            "选择功能",
            ["📊 股票分析", "📈 技术指标", "🎯 AI建议", "📋 投资组合"]
        )
        
        st.markdown("---")
        
        # 股票选择
        symbol = st.text_input("输入股票代码", value="AAPL").upper()
        
        # 时间周期选择
        period = st.selectbox(
            "选择时间周期",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        # 分析按钮
        analyze_button = st.button("🔍 开始分析", type="primary")
        
        st.markdown("---")
        st.markdown("### 热门股票")
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        for stock in popular_stocks:
            if st.button(stock, key=f"quick_{stock}"):
                symbol = stock
                analyze_button = True
    
    # 主界面标题
    st.title("📈 QuantGPT - AI量化交易平台")
    st.markdown("使用人工智能和量化分析技术，为您提供专业的投资建议")
    
    # 初始化分析器
    analyzer = SimpleQuantAnalyzer()
    
    # 分析逻辑
    if analyze_button or symbol:
        with st.spinner(f"正在分析 {symbol}..."):
            # 获取数据
            data, info = analyzer.get_stock_data(symbol, period)
            
            if data is not None:
                # 计算技术指标
                data_with_indicators = analyzer.calculate_technical_indicators(data)
                
                # 显示不同页面内容
                if "股票分析" in page:
                    display_stock_analysis(symbol, data_with_indicators, info, analyzer)
                elif "技术指标" in page:
                    display_technical_analysis(symbol, data_with_indicators)
                elif "AI建议" in page:
                    display_ai_recommendations(symbol, data_with_indicators, info, analyzer)
                elif "投资组合" in page:
                    display_portfolio_analysis(symbol, data_with_indicators, info)
            else:
                st.error(f"无法获取 {symbol} 的数据，请检查股票代码是否正确")
    else:
        # 显示欢迎页面
        display_welcome_page()

def display_welcome_page():
    """显示欢迎页面"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 🤖 AI分析\n使用先进的AI模型分析市场情绪和趋势")
    
    with col2:
        st.success("### 📊 技术指标\n全面的技术分析工具和图表")
    
    with col3:
        st.warning("### 💰 策略回测\n测试和优化您的交易策略")
    
    st.markdown("---")
    st.markdown("### 快速开始")
    st.markdown("1. 在左侧输入股票代码（如 AAPL, GOOGL, TSLA）")
    st.markdown("2. 选择分析时间周期")
    st.markdown("3. 点击 **开始分析** 按钮")
    
    # 市场概览
    st.markdown("### 📈 今日市场")
    market_indices = {
        "^GSPC": "S&P 500",
        "^DJI": "道琼斯",
        "^IXIC": "纳斯达克",
        "^VIX": "VIX恐慌指数"
    }
    
    cols = st.columns(4)
    for i, (ticker, name) in enumerate(market_indices.items()):
        with cols[i]:
            try:
                data = yf.Ticker(ticker).history(period="1d")
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    prev = data['Open'].iloc[0]
                    change = (current - prev) / prev * 100
                    delta_color = "inverse" if change >= 0 else "normal"
                    st.metric(name, f"{current:.2f}", f"{change:.2f}%", delta_color=delta_color)
            except:
                st.metric(name, "N/A", "N/A")

def display_stock_analysis(symbol, data, info, analyzer):
    """显示股票分析页面"""
    st.header(f"📊 {symbol} 股票分析")
    
    # 基本信息
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
    change = (current_price - prev_close) / prev_close * 100
    
    with col1:
        st.metric("当前价格", f"${current_price:.2f}", f"{change:.2f}%")
    
    with col2:
        volume = data['Volume'].iloc[-1]
        st.metric("成交量", f"{volume:,.0f}", "")
    
    with col3:
        if info:
            pe = info.get('trailingPE', 'N/A')
            if pe != 'N/A':
                st.metric("市盈率", f"{pe:.2f}", "")
            else:
                st.metric("市盈率", "N/A", "")
    
    with col4:
        if info:
            market_cap = info.get('marketCap', 0)
            if market_cap:
                st.metric("市值", f"${market_cap/1e9:.1f}B", "")
            else:
                st.metric("市值", "N/A", "")
    
    # 价格图表
    st.subheader("价格走势")
    fig = go.Figure()
    
    # 添加K线图
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
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], 
                                name='SMA 20', line=dict(color='orange')))
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], 
                                name='SMA 50', line=dict(color='blue')))
    
    fig.update_layout(
        title=f"{symbol} 价格走势图",
        yaxis_title="价格 ($)",
        xaxis_title="日期",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 成交量图
    st.subheader("成交量")
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=data.index, y=data['Volume'], name='成交量'))
    fig_volume.update_layout(
        title="成交量走势",
        yaxis_title="成交量",
        xaxis_title="日期",
        height=300,
        template="plotly_white"
    )
    st.plotly_chart(fig_volume, use_container_width=True)

def display_technical_analysis(symbol, data):
    """显示技术分析页面"""
    st.header(f"📈 {symbol} 技术分析")
    
    # 技术指标面板
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RSI指标")
        if 'RSI' in data.columns:
            current_rsi = data['RSI'].iloc[-1]
            
            # RSI仪表盘
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_rsi,
                title={'text': "RSI (14)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            if current_rsi < 30:
                st.success("超卖区域 - 可能的买入机会")
            elif current_rsi > 70:
                st.warning("超买区域 - 可能的卖出机会")
            else:
                st.info("中性区域")
    
    with col2:
        st.subheader("布林带")
        if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            current_price = data['Close'].iloc[-1]
            bb_upper = data['BB_Upper'].iloc[-1]
            bb_lower = data['BB_Lower'].iloc[-1]
            bb_middle = data['BB_Middle'].iloc[-1]
            
            position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
            
            st.metric("当前价格", f"${current_price:.2f}")
            st.metric("上轨", f"${bb_upper:.2f}")
            st.metric("中轨", f"${bb_middle:.2f}")
            st.metric("下轨", f"${bb_lower:.2f}")
            
            if position < 20:
                st.success("接近下轨 - 可能反弹")
            elif position > 80:
                st.warning("接近上轨 - 可能回调")
            else:
                st.info("在布林带中间区域")
    
    # MACD图表
    st.subheader("MACD指标")
    if 'MACD' in data.columns and 'Signal' in data.columns:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                                name='MACD', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], 
                                name='Signal', line=dict(color='red')))
        
        histogram = data['MACD'] - data['Signal']
        colors = ['green' if val >= 0 else 'red' for val in histogram]
        fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram',
                           marker_color=colors))
        
        fig.update_layout(
            title="MACD (12, 26, 9)",
            yaxis_title="值",
            xaxis_title="日期",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_ai_recommendations(symbol, data, info, analyzer):
    """显示AI建议页面"""
    st.header(f"🎯 {symbol} AI智能建议")
    
    # 生成AI建议
    recommendation, reason, confidence = analyzer.generate_ai_recommendation(data, info)
    
    # 显示建议卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 投资建议")
        st.markdown(f"## {recommendation}")
    
    with col2:
        st.markdown("### 置信度")
        st.progress(confidence)
        st.markdown(f"**{confidence*100:.1f}%**")
    
    with col3:
        st.markdown("### 分析依据")
        st.info(reason)
    
    st.markdown("---")
    
    # 详细分析
    st.subheader("📊 详细分析")
    
    analysis_data = []
    
    # 技术面分析
    if not data.empty:
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data else None
        sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data else None
        rsi = data['RSI'].iloc[-1] if 'RSI' in data else None
        
        if sma_20 and sma_50:
            if current_price > sma_20 > sma_50:
                analysis_data.append(("均线系统", "✅ 多头排列", "positive"))
            elif current_price < sma_20 < sma_50:
                analysis_data.append(("均线系统", "❌ 空头排列", "negative"))
            else:
                analysis_data.append(("均线系统", "⚪ 震荡整理", "neutral"))
        
        if rsi:
            if rsi < 30:
                analysis_data.append(("RSI指标", "✅ 超卖区域", "positive"))
            elif rsi > 70:
                analysis_data.append(("RSI指标", "❌ 超买区域", "negative"))
            else:
                analysis_data.append(("RSI指标", f"⚪ 中性 ({rsi:.1f})", "neutral"))
    
    # 基本面分析
    if info:
        pe = info.get('trailingPE')
        if pe:
            if pe < 15:
                analysis_data.append(("市盈率", f"✅ 低估 (PE={pe:.1f})", "positive"))
            elif pe > 30:
                analysis_data.append(("市盈率", f"❌ 高估 (PE={pe:.1f})", "negative"))
            else:
                analysis_data.append(("市盈率", f"⚪ 合理 (PE={pe:.1f})", "neutral"))
    
    # 显示分析结果
    for indicator, status, sentiment in analysis_data:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**{indicator}:**")
        with col2:
            if sentiment == "positive":
                st.success(status)
            elif sentiment == "negative":
                st.error(status)
            else:
                st.info(status)
    
    # 风险提示
    st.markdown("---")
    st.warning("⚠️ **风险提示**: 以上建议仅供参考，投资有风险，入市需谨慎。请结合自身情况做出投资决策。")

def display_portfolio_analysis(symbol, data, info):
    """显示投资组合分析"""
    st.header("📋 投资组合分析")
    
    # 模拟投资组合
    st.subheader("模拟投资")
    
    col1, col2 = st.columns(2)
    
    with col1:
        investment = st.number_input("投资金额 ($)", min_value=100, value=10000, step=100)
        shares = int(investment / data['Close'].iloc[-1])
        st.info(f"可购买 **{shares}** 股")
    
    with col2:
        period_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        potential_return = investment * period_return
        st.metric("期间收益率", f"{period_return*100:.2f}%", f"${potential_return:.2f}")
    
    # 风险分析
    st.subheader("风险指标")
    
    returns = data['Close'].pct_change().dropna()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        volatility = returns.std() * np.sqrt(252)
        st.metric("年化波动率", f"{volatility*100:.2f}%")
    
    with col2:
        max_drawdown = (data['Close'] / data['Close'].cummax() - 1).min()
        st.metric("最大回撤", f"{max_drawdown*100:.2f}%")
    
    with col3:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        st.metric("夏普比率", f"{sharpe:.2f}")
    
    # 持仓建议
    st.subheader("持仓建议")
    
    risk_level = st.select_slider(
        "您的风险承受能力",
        options=["保守", "稳健", "平衡", "积极", "激进"],
        value="平衡"
    )
    
    position_size = {
        "保守": 5,
        "稳健": 10,
        "平衡": 15,
        "积极": 20,
        "激进": 30
    }
    
    recommended_position = position_size[risk_level]
    
    st.info(f"根据您的风险偏好，建议 {symbol} 占投资组合的 **{recommended_position}%**")
    
    # 分散投资建议
    st.subheader("分散投资建议")
    
    if info:
        sector = info.get('sector', 'Unknown')
        st.markdown(f"**{symbol}** 属于 **{sector}** 板块")
        st.markdown("建议配置其他板块的股票以分散风险：")
        
        other_sectors = ["科技", "医疗", "金融", "消费", "能源", "工业"]
        other_sectors = [s for s in other_sectors if s != sector]
        
        cols = st.columns(len(other_sectors[:3]))
        for i, sector in enumerate(other_sectors[:3]):
            with cols[i]:
                st.info(f"考虑 {sector} 板块")

# ===================================
# 运行应用
# ===================================

if __name__ == "__main__":
    main()
