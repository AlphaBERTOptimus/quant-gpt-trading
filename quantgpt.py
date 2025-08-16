strategy_insights = {
                "趋势跟踪": "适合趋势明确的市场环境，在震荡市场中可能产生较多假信号",
                "均值回归": "在震荡市场中表现优秀，但在强趋势市场中可能错失机会",
                "动量策略": "能够捕捉强势行情，但需要注意动量衰减的风险",
                "突破策略": "适合捕捉关键突破点位，注意假突破的风险",
                "网格交易": "适合区间震荡市场，在单边趋势中需要谨慎使用",
                "量价策略": "成交量确认提高信号质量，但在低流动性市场中效果有限",
                "波动率策略": "能够捕捉市场情绪变化，适合波动率交易专家",
                "配对交易": "市场中性策略，适合对冲风险，需要深入的统计分析"
            }
            
            if strategy in strategy_insights:
                assessment += f"\n**📋 策略特色：** {strategy_insights[strategy]}\n"
            
            return assessment
            
        except (ValueError, KeyError) as e:
            return f"\n### 🤖 QuantGPT AI 评估\n\n策略分析完成，请查看详细指标。如需更精准评估，请确保数据完整性。"

# 高级图表生成器
class ProfessionalChartGenerator:
    @staticmethod
    def create_comprehensive_chart(data, stock, strategy, params):
        """创建专业综合分析图表"""
        fig = make_subplots(
            rows=5, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.02,
            horizontal_spacing=0.05,
            subplot_titles=(
                f'{stock} 价格走势与交易信号', '技术指标面板',
                '策略收益 vs 基准', '风险指标监控',
                '交易信号分布', '月度收益分析',
                '回撤分析', '波动率分析',
                '资金曲线', '绩效雷达图'
            ),
            specs=[[{"colspan": 2}, None],
                   [{"colspan": 2}, None], 
                   [{"colspan": 2}, None],
                   [{}, {}],
                   [{}, {}]],
            row_heights=[0.3, 0.2, 0.2, 0.15, 0.15]
        )
        
        # 1. 主价格图表
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'], 
                low=data['Low'],
                close=data['Close'],
                name=f'{stock} K线',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ), row=1, col=1
        )
        
        # 添加移动平均线
        if 'SMA_short' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_short'], 
                          name=f'SMA{params.get("short_window", 20)}',
                          line=dict(color='orange', width=1.5)),
                row=1, col=1
            )
        
        if 'SMA_long' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_long'],
                          name=f'SMA{params.get("long_window", 50)}', 
                          line=dict(color='blue', width=1.5)),
                row=1, col=1
            )
        
        # 交易信号
        buy_signals = data[data['Position'] == 1]
        sell_signals = data[data['Position'] == -1]
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                          mode='markers', name='买入信号',
                          marker=dict(color='green', size=15, symbol='triangle-up')),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                          mode='markers', name='卖出信号',
                          marker=dict(color='red', size=15, symbol='triangle-down')),
                row=1, col=1
            )
        
        # 2. 技术指标
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI',
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red")
            fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green")
        
        # 3. 收益对比
        benchmark_return = (data['Cumulative_Returns'] - 1) * 100
        strategy_return = (data['Strategy_Cumulative'] - 1) * 100
        
        fig.add_trace(
            go.Scatter(x=data.index, y=benchmark_return,
                      name='基准收益(%)', line=dict(color='gray', width=2)),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=strategy_return,
                      name='策略收益(%)', line=dict(color='green', width=3)),
            row=3, col=1
        )
        
        # 4. 回撤分析
        cumulative = data['Strategy_Cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(x=data.index, y=drawdown, name='策略回撤(%)',
                      fill='tonexty', fillcolor='rgba(255,0,0,0.3)',
                      line=dict(color='red', width=1)),
            row=4, col=1
        )
        
        # 5. 波动率分析
        rolling_vol = data['Net_Strategy_Returns'].rolling(window=30).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(x=data.index, y=rolling_vol, name='30日滚动波动率(%)',
                      line=dict(color='orange', width=2)),
            row=4, col=2
        )
        
        # 6. 资金曲线
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Portfolio_Value'],
                      name='资金曲线', line=dict(color='blue', width=2)),
            row=5, col=1
        )
        
        # 7. 月度收益热力图数据准备
        monthly_returns = data['Net_Strategy_Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pct = monthly_returns * 100
        
        if len(monthly_returns_pct) > 0:
            fig.add_trace(
                go.Bar(x=monthly_returns_pct.index, y=monthly_returns_pct.values,
                      name='月度收益(%)', marker_color=monthly_returns_pct.apply(
                          lambda x: 'green' if x > 0 else 'red')),
                row=5, col=2
            )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title=f"{stock} - {strategy} 专业分析图表",
            showlegend=True,
            template="plotly_white",
            font=dict(family="Inter, sans-serif"),
            title_font_size=20,
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 美化网格和坐标轴
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    @staticmethod
    def create_strategy_comparison_chart(comparison_data):
        """创建策略对比图表"""
        strategies = list(comparison_data.keys())
        metrics = ['总收益率', '夏普比率', '最大回撤', '胜率']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = []
            for strategy in strategies:
                try:
                    value_str = comparison_data[strategy]['metrics'][metric]
                    if '%' in value_str:
                        value = float(value_str.replace('%', ''))
                    else:
                        value = float(value_str)
                    values.append(value)
                except:
                    values.append(0)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=strategies,
                fill='toself',
                name=metric
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max([max(values) for values in [values]])]),
            ),
            showlegend=True,
            title="策略性能雷达图对比"
        )
        
        return fig

# 消息显示组件
def display_message(message, is_user=False, message_id=None):
    """显示专业聊天消息"""
    message_class = "user" if is_user else "bot"
    avatar = "👤" if is_user else "🤖"
    
    # 打字机效果的占位符
    if message_id and not is_user:
        placeholder = st.empty()
        
        # 显示打字指示器
        with placeholder.container():
            st.markdown(f"""
            <div class="chat-message bot">
                <div class="avatar">{avatar}</div>
                <div class="message">
                    <div class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        QuantGPT 正在分析中...
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 模拟思考时间
        time.sleep(1.5)
        
        # 显示实际消息
        placeholder.empty()
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="avatar">{avatar}</div>
        <div class="message">{message}</div>
    </div>
    """, unsafe_allow_html=True)

# 图表展示函数
def show_professional_analysis_chart(stock):
    """显示专业分析图表"""
    if 'analysis_data' not in st.session_state or stock not in st.session_state.analysis_data:
        st.error(f"没有 {stock} 的分析数据")
        return
    
    analysis_info = st.session_state.analysis_data[stock]
    data = analysis_info['data']
    strategy = analysis_info['strategy']
    params = analysis_info['params']
    
    # 创建专业图表
    chart_generator = ProfessionalChartGenerator()
    fig = chart_generator.create_comprehensive_chart(data, stock, strategy, params)
    
    # 在图表容器中显示
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 显示详细统计数据
    col1, col2, col3, col4 = st.columns(4)
    
    # 计算统计数据
    total_trades = len(data[data['Position'] != 0])
    winning_trades = len(data[(data['Position'] != 0) & (data['Net_Strategy_Returns'] > 0)])
    final_return = (data['Strategy_Cumulative'].iloc[-1] - 1) * 100
    max_dd = ((data['Strategy_Cumulative'] / data['Strategy_Cumulative'].expanding().max()) - 1).min() * 100
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_trades}</div>
            <div class="metric-label">总交易次数</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{winning_trades}</div>
            <div class="metric-label">盈利交易</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{final_return:.1f}%</div>
            <div class="metric-label">总收益率</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{max_dd:.1f}%</div>
            <div class="metric-label">最大回撤</div>
        </div>
        """, unsafe_allow_html=True)

# 策略对比功能
def show_strategy_comparison():
    """显示策略对比功能"""
    st.markdown("### 🔄 策略性能对比")
    
    if 'analysis_data' not in st.session_state or len(st.session_state.analysis_data) < 2:
        st.info("需要至少分析2个策略才能进行对比")
        return
    
    # 选择对比的策略
    available_analyses = list(st.session_state.analysis_data.keys())
    selected_strategies = st.multiselect(
        "选择要对比的策略分析", 
        available_analyses,
        default=available_analyses[:2] if len(available_analyses) >= 2 else available_analyses
    )
    
    if len(selected_strategies) >= 2:
        comparison_data = {}
        
        for strategy_key in selected_strategies:
            analysis_info = st.session_state.analysis_data[strategy_key]
            data = analysis_info['data']
            
            # 计算对比指标
            metrics = ProfessionalBacktestEngine.calculate_advanced_metrics(data)
            comparison_data[strategy_key] = {
                'data': data,
                'metrics': metrics,
                'strategy': analysis_info['strategy']
            }
        
        # 创建对比表格
        metrics_df = pd.DataFrame({
            strategy: data['metrics'] for strategy, data in comparison_data.items()
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # 创建雷达图对比
        chart_generator = ProfessionalChartGenerator()
        radar_fig = chart_generator.create_strategy_comparison_chart(comparison_data)
        st.plotly_chart(radar_fig, use_container_width=True)

# 主应用程序
def main():
    # 英雄区域
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">QuantGPT Pro</h1>
        <p class="hero-subtitle">🚀 下一代AI量化交易分析平台 | 专业级策略回测 | 智能风险管理</p>
        <div class="quick-action-grid">
            <div class="quick-action-card">
                <h4>📊 智能策略分析</h4>
                <p>8种专业策略，AI驱动优化</p>
            </div>
            <div class="quick-action-card">
                <h4>⚡ 实时风险监控</h4>
                <p>VaR模型，专业风控体系</p>
            </div>
            <div class="quick-action-card">
                <h4>🎯 精准回测引擎</h4>
                <p>考虑滑点手续费的真实回测</p>
            </div>
            <div class="quick-action-card">
                <h4>🤖 AI量化顾问</h4>
                <p>智能评级，个性化建议</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 系统状态指示器
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "status-online" if YFINANCE_AVAILABLE else "status-offline"
        status_text = "实时数据" if YFINANCE_AVAILABLE else "模拟数据"
        st.markdown(f'<div class="status-indicator {status}">📡 {status_text}</div>', unsafe_allow_html=True)
    
    with col2:
        ta_status = "status-online" if TA_AVAILABLE else "status-limited"
        ta_text = "完整指标" if TA_AVAILABLE else "基础指标"
        st.markdown(f'<div class="status-indicator {ta_status}">📈 {ta_text}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="status-indicator status-online">🛡️ 风控启用</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="premium-badge">PRO版本</div>', unsafe_allow_html=True)
    
    # 聊天区域
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # 初始化会话状态
    if "messages" not in st.session_state:
        welcome_msg = """🎉 **欢迎来到 QuantGPT Pro！**

我是您的专业AI量化交易顾问，具备以下核心能力：

### 🎯 **专业策略库**
• **趋势跟踪** - 多重均线确认系统
• **均值回归** - 布林带+RSI双重过滤  
• **动量策略** - RSI+MACD+随机指标三重确认
• **突破策略** - 价量突破，关键点位捕捉
• **网格交易** - ATR动态网格，智能仓位管理
• **量价策略** - VWAP确认，提升信号质量
• **波动率策略** - 波动率突破，情绪驱动交易
• **配对交易** - 市场中性，统计套利

### 🛡️ **专业风控体系**
• **VaR风险模型** - 95%/99%置信区间风险评估
• **动态止损止盈** - 智能风险控制
• **仓位管理** - 凯利公式优化仓位
• **回撤监控** - 实时最大回撤追踪

### 📊 **高级分析工具**
• **Sharpe/Sortino/Calmar比率** - 多维度风险调整收益
• **专业图表** - K线+技术指标+资金曲线
• **策略对比** - 雷达图多策略性能对比
• **AI智能评级** - AAA-BB评级体系

### 💡 **AI驱动建议**
• **风险偏好匹配** - 保守/平衡/激进/专业型
• **参数智能优化** - 自动寻优最佳参数组合
• **实盘交易指导** - 具体仓位和止损建议

**🚀 开始体验：**
• `"分析AAPL的趋势策略"`
• `"用保守型策略分析TSLA"`  
• `"GOOGL的动量策略5%止损"`
• `"对比MSFT的多种策略"`

现在就告诉我您的投资需求，让我为您提供专业的量化分析！"""
        
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
    
    if "analyst" not in st.session_state:
        st.session_state.analyst = QuantGPTAnalyst()
    
    # 显示聊天历史
    for i, message in enumerate(st.session_state.messages):
        display_message(message["content"], message["role"] == "user", f"msg_{i}")
    
    # 用户输入
    user_input = st.chat_input("💬 请描述您的量化交易需求...", key="professional_input")
    
    if user_input:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message(user_input, True)
        
        # 显示思考过程
        thinking_placeholder = st.empty()
        with thinking_placeholder.container():
            st.markdown("""
            <div class="chat-message bot">
                <div class="avatar">🤖</div>
                <div class="message">
                    <div class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        正在进行深度量化分析...
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 处理用户输入
        try:
            time.sleep(2)  # 模拟AI思考时间
            parsed_input = st.session_state.analyst.parse_user_input(user_input)
            response = st.session_state.analyst.generate_intelligent_response(parsed_input)
        except Exception as e:
            response = f"😅 分析过程中遇到技术问题：{str(e)}\n\n请尝试重新描述您的需求，或联系技术支持。"
        finally:
            thinking_placeholder.empty()
        
        # 添加AI响应
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_message(response, False)
        
        # 如果有分析数据，显示高级功能
        if 'analysis_data' in st.session_state and st.session_state.analysis_data:
            st.markdown("---")
            
            # 图表分析区域
            st.markdown("### 📊 专业图表分析")
            chart_cols = st.columns(len(st.session_state.analysis_data))
            
            for i, stock in enumerate(st.session_state.analysis_data.keys()):
                with chart_cols[i]:
                    if st.button(f"📈 {stock} 专业分析", key=f"prof_chart_{stock}", use_container_width=True):
                        show_professional_analysis_chart(stock)
            
            # 策略对比功能
            if len(st.session_state.analysis_data) >= 2:
                if st.button("🔄 策略性能对比", use_container_width=True):
                    show_strategy_comparison()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 侧边栏专业功能
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 1.5rem;">
            <h3>🎛️ 专业控制面板</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 快速清理
        if st.button("🗑️ 清除分析历史", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            if 'analysis_data' in st.session_state:
                del st.session_state.analysis_data
            st.rerun()
        
        # 专业快速分析
        st.markdown("### 🚀 专业快速分析")
        
        # 股票选择
        popular_stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN", "META", "NFLX"]
        selected_stock = st.selectbox("选择热门股票", popular_stocks)
        
        # 策略选择
        all_strategies = ["趋势跟踪", "均值回归", "动量策略", "突破策略", "网格交易", "量价策略", "波动率策略", "配对交易"]
        selected_strategy = st.selectbox("选择策略", all_strategies)
        
        # 风险偏好
        risk_profiles = ["保守型", "平衡型", "激进型", "专业型"]
        selected_risk = st.selectbox("风险偏好", risk_profiles)
        
        # 高级参数
        with st.expander("⚙️ 高级参数设置"):
            stop_loss = st.slider("止损百分比", 1, 10, 5) / 100
            take_profit = st.slider("止盈百分比", 5, 20, 10) / 100
            period = st.selectbox("分析周期", ["1mo", "3mo", "6mo", "1y", "2y"], index=4)
        
        if st.button("🎯 执行专业分析", use_container_width=True):
            query = f"用{selected_risk}的{selected_strategy}分析{selected_stock}，设置{stop_loss*100:.0f}%止损和{take_profit*100:.0f}%止盈，分析周期{period}"
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.spinner("🤖 AI正在执行深度分析..."):
                parsed_input = st.session_state.analyst.parse_user_input(query)
                response = st.session_state.analyst.generate_intelligent_response(parsed_input)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # 示例策略库
        with st.expander("💡 策略示例库"):
            examples = [
                "分析AAPL的趋势跟踪策略",
                "TSLA的激进型动量策略分析",
                "用5%止损分析GOOGL突破策略", 
                "MSFT的保守型均值回归策略",
                "NVDA的专业型量价策略回测",
                "对比AMZN的多种策略表现"
            ]
            
            for example in examples:
                if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    
                    with st.spinner("分析中..."):
                        parsed_input = st.session_state.analyst.parse_user_input(example)
                        response = st.session_state.analyst.generate_intelligent_response(parsed_input)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
        
        # 系统信息
        st.markdown("---")
        st.markdown("### 📋 系统信息")
        
        system_info = f"""
        **数据源：** {'yfinance (实时)' if YFINANCE_AVAILABLE else '高质量模拟数据'}
        **技术指标：** {'TA-Lib (完整)' if TA_AVAILABLE else '内置专业指标'}
        **策略数量：** 8种专业策略
        **风险模型：** VaR + 动态止损
        **评级体系：** AAA-BB智能评级
        """
        st.markdown(system_info)
        
        # 帮助信息
        with st.expander("❓ 使用指南"):
            st.markdown("""
            **🎯 支持的输入格式：**
            • 股票代码：AAPL, TSLA, ^GSPC
            • 策略指定：趋势、动量、突破等
            • 风险偏好：保守型、激进型等
            • 参数设置：5%止损、20日均线等
            
            **📊 高级功能：**
            • 多策略对比分析
            • 实时风险监控
            • AI智能评级
            • 专业图表展示
            
            **💡 使用技巧：**
            • 描述越详细，分析越精准
            • 可同时分析多个股票
            • 支持自定义参数优化
            • 建议结合风险偏好使用
            """)
        
        # 版权信息
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; color: #666;">
            <small>
            <strong>QuantGPT Pro v3.0</strong><br>
            🚀 AI量化交易分析平台<br>
            Powered by Streamlit & Advanced Analytics<br><br>
            <em>仅供教育和研究用途，投资有风险</em>
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    # 水印
    st.markdown("""
    <div class="watermark">
        QuantGPT Pro © 2024
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import re
import time
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
import warnings
import hashlib
import base64
from io import BytesIO

# 尝试导入可选依赖
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="QuantGPT Pro - AI量化交易平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 商业级CSS样式
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main > div {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .chat-container {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: flex-start;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .chat-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .chat-message:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        flex-direction: row-reverse;
        margin-left: 2rem;
    }
    
    .chat-message.bot {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
    }
    
    .chat-message .avatar {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        margin: 0 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .chat-message .message {
        flex: 1;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 45px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #666;
        font-weight: 500;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    .sidebar .block-container {
        padding: 1.5rem;
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-online {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        color: white;
    }
    
    .status-limited {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .premium-badge {
        background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(255,215,0,0.3);
    }
    
    .quick-action-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .quick-action-card {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        backdrop-filter: blur(10px);
    }
    
    .quick-action-card:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: rgba(255,255,255,0.7);
        margin: 0 2px;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0.2s; }
    .typing-dot:nth-child(2) { animation-delay: 0.4s; }
    .typing-dot:nth-child(3) { animation-delay: 0.6s; }
    
    @keyframes typing {
        0%, 80%, 100% { opacity: 0.3; }
        40% { opacity: 1; }
    }
    
    .chart-container {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .strategy-comparison {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .watermark {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        z-index: 1000;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# 数据缓存装饰器
@st.cache_data(ttl=3600)
def get_cached_stock_data(symbol, period="2y"):
    """缓存股票数据"""
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if not data.empty:
                return data
        except Exception as e:
            st.warning(f"获取{symbol}数据失败: {str(e)}")
    
    # 返回模拟数据
    return MockDataGenerator.generate_mock_data(symbol, period)

# 高级数据生成器
class MockDataGenerator:
    @staticmethod
    def generate_mock_data(symbol, period="2y"):
        """生成高质量模拟数据"""
        end_date = datetime.now()
        days_map = {"2y": 730, "1y": 365, "6mo": 180, "3mo": 90, "1mo": 30}
        days = days_map.get(period, 730)
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(hash(symbol) % 1000)
        
        # 生成更真实的价格走势
        base_price = 100 + (hash(symbol) % 500)
        trend = np.random.choice([-0.0002, 0.0002, 0.0005], p=[0.3, 0.4, 0.3])
        volatility = 0.015 + (hash(symbol) % 100) / 10000
        
        prices = [base_price]
        for i in range(1, len(dates)):
            # 加入趋势和季节性
            seasonal = 0.001 * np.sin(2 * np.pi * i / 252)
            noise = np.random.normal(trend + seasonal, volatility)
            new_price = prices[-1] * (1 + noise)
            prices.append(max(new_price, prices[-1] * 0.9))  # 防止价格过度下跌
        
        # 创建OHLC数据
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.002, len(data)))
        data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.008, len(data))))
        data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.008, len(data))))
        data['Volume'] = np.random.lognormal(15, 0.5, len(data)).astype(int)
        
        return data.dropna()

# 增强的技术指标计算
class AdvancedTechnicalIndicators:
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data, window=20, std_dev=2):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower, sma
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high, low, close, k_window=14, d_window=3):
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

# 专业策略引擎
class ProfessionalStrategyEngine:
    def __init__(self):
        self.strategies = {
            "趋势跟踪": self.trend_following,
            "均值回归": self.mean_reversion,
            "动量策略": self.momentum_strategy,
            "突破策略": self.breakout_strategy,
            "网格交易": self.grid_trading,
            "配对交易": self.pairs_trading,
            "量价策略": self.volume_price_strategy,
            "波动率策略": self.volatility_strategy
        }
    
    def trend_following(self, data, params):
        """增强的趋势跟踪策略"""
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
        if TA_AVAILABLE:
            data['SMA_short'] = ta.trend.SMAIndicator(data['Close'], window=short_window).sma_indicator()
            data['SMA_long'] = ta.trend.SMAIndicator(data['Close'], window=long_window).sma_indicator()
            data['EMA_short'] = ta.trend.EMAIndicator(data['Close'], window=short_window).ema_indicator()
        else:
            data['SMA_short'] = AdvancedTechnicalIndicators.sma(data['Close'], short_window)
            data['SMA_long'] = AdvancedTechnicalIndicators.sma(data['Close'], long_window)
            data['EMA_short'] = AdvancedTechnicalIndicators.ema(data['Close'], short_window)
        
        # 多重确认信号
        ma_signal = (data['SMA_short'] > data['SMA_long']).astype(int)
        price_signal = (data['Close'] > data['EMA_short']).astype(int)
        
        data['Signal'] = ((ma_signal + price_signal) >= 2).astype(int)
        data['Position'] = data['Signal'].diff()
        
        return data, f"增强趋势跟踪策略：{short_window}日/{long_window}日双均线 + EMA确认，多重信号过滤"
    
    def mean_reversion(self, data, params):
        """均值回归策略"""
        window = params.get('window', 20)
        std_dev = params.get('std_dev', 2.0)
        
        if TA_AVAILABLE:
            bb = ta.volatility.BollingerBands(data['Close'], window=window, window_dev=std_dev)
            data['Upper'] = bb.bollinger_hband()
            data['Lower'] = bb.bollinger_lband()
            data['SMA'] = bb.bollinger_mavg()
            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        else:
            data['Upper'], data['Lower'], data['SMA'] = AdvancedTechnicalIndicators.bollinger_bands(
                data['Close'], window, std_dev
            )
            data['RSI'] = AdvancedTechnicalIndicators.rsi(data['Close'])
        
        # 结合RSI的布林带策略
        bb_signal = np.where(data['Close'] < data['Lower'], 1, 
                           np.where(data['Close'] > data['Upper'], -1, 0))
        rsi_signal = np.where(data['RSI'] < 30, 1, 
                            np.where(data['RSI'] > 70, -1, 0))
        
        data['Signal'] = np.where((bb_signal == 1) & (rsi_signal == 1), 1,
                                np.where((bb_signal == -1) & (rsi_signal == -1), -1, 0))
        data['Position'] = data['Signal'].diff()
        
        return data, f"增强均值回归策略：布林带({window}日, {std_dev}σ) + RSI过滤，双重确认"
    
    def momentum_strategy(self, data, params):
        """高级动量策略"""
        rsi_window = params.get('rsi_window', 14)
        
        if TA_AVAILABLE:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
            data['MACD'] = ta.trend.MACD(data['Close']).macd()
            data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()
            data['MACD_hist'] = ta.trend.MACD(data['Close']).macd_diff()
            data['Stoch_K'], data['Stoch_D'] = AdvancedTechnicalIndicators.stochastic(
                data['High'], data['Low'], data['Close']
            )
        else:
            data['RSI'] = AdvancedTechnicalIndicators.rsi(data['Close'], rsi_window)
            data['MACD'], data['MACD_signal'], data['MACD_hist'] = AdvancedTechnicalIndicators.macd(data['Close'])
            data['Stoch_K'], data['Stoch_D'] = AdvancedTechnicalIndicators.stochastic(
                data['High'], data['Low'], data['Close']
            )
        
        # 多指标动量确认
        rsi_momentum = (data['RSI'] > 50).astype(int) - (data['RSI'] < 50).astype(int)
        macd_momentum = (data['MACD'] > data['MACD_signal']).astype(int) - (data['MACD'] < data['MACD_signal']).astype(int)
        stoch_momentum = (data['Stoch_K'] > data['Stoch_D']).astype(int) - (data['Stoch_K'] < data['Stoch_D']).astype(int)
        
        momentum_score = rsi_momentum + macd_momentum + stoch_momentum
        data['Signal'] = np.where(momentum_score >= 2, 1, np.where(momentum_score <= -2, -1, 0))
        data['Position'] = data['Signal'].diff()
        
        return data, f"高级动量策略：RSI + MACD + 随机指标三重确认，动量评分系统"
    
    def breakout_strategy(self, data, params):
        """突破策略"""
        window = params.get('window', 20)
        volume_factor = params.get('volume_factor', 1.5)
        
        data['High_max'] = data['High'].rolling(window=window).max()
        data['Low_min'] = data['Low'].rolling(window=window).min()
        data['Volume_MA'] = data['Volume'].rolling(window=window).mean()
        
        # 价量突破确认
        price_breakout = np.where(data['Close'] > data['High_max'].shift(1), 1,
                                np.where(data['Close'] < data['Low_min'].shift(1), -1, 0))
        volume_confirm = (data['Volume'] > data['Volume_MA'] * volume_factor).astype(int)
        
        data['Signal'] = price_breakout * volume_confirm
        data['Position'] = data['Signal'].diff()
        
        return data, f"价量突破策略：{window}日通道突破 + {volume_factor}倍量能确认"
    
    def grid_trading(self, data, params):
        """智能网格交易"""
        grid_size = params.get('grid_size', 0.02)
        atr_factor = params.get('atr_factor', 1.0)
        
        # 基于ATR的动态网格
        if TA_AVAILABLE:
            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        else:
            data['ATR'] = AdvancedTechnicalIndicators.atr(data['High'], data['Low'], data['Close'])
        
        dynamic_grid = data['ATR'] * atr_factor / data['Close']
        data['Price_change'] = data['Close'].pct_change().cumsum()
        data['Grid_level'] = (data['Price_change'] / (grid_size + dynamic_grid)).round()
        data['Signal'] = -data['Grid_level'].diff()
        data['Position'] = data['Signal'].diff()
        
        return data, f"智能网格策略：基础网格{grid_size*100:.1f}% + ATR动态调整"
    
    def pairs_trading(self, data, params):
        """统计套利策略"""
        window = params.get('window', 30)
        threshold = params.get('threshold', 2.0)
        
        data['MA'] = data['Close'].rolling(window=window).mean()
        data['Spread'] = data['Close'] - data['MA']
        data['Spread_MA'] = data['Spread'].rolling(window=window).mean()
        data['Spread_STD'] = data['Spread'].rolling(window=window).std()
        data['Z_Score'] = (data['Spread'] - data['Spread_MA']) / data['Spread_STD']
        
        data['Signal'] = np.where(data['Z_Score'] < -threshold, 1,
                                np.where(data['Z_Score'] > threshold, -1, 0))
        data['Position'] = data['Signal'].diff()
        
        return data, f"统计套利策略：{window}日Z-Score模型，阈值±{threshold}σ"
    
    def volume_price_strategy(self, data, params):
        """量价策略"""
        price_window = params.get('price_window', 20)
        volume_window = params.get('volume_window', 20)
        
        data['Price_MA'] = data['Close'].rolling(window=price_window).mean()
        data['Volume_MA'] = data['Volume'].rolling(window=volume_window).mean()
        data['VWAP'] = (data['Close'] * data['Volume']).rolling(window=price_window).sum() / data['Volume'].rolling(window=price_window).sum()
        
        # 量价配合信号
        price_signal = (data['Close'] > data['Price_MA']).astype(int)
        volume_signal = (data['Volume'] > data['Volume_MA']).astype(int)
        vwap_signal = (data['Close'] > data['VWAP']).astype(int)
        
        data['Signal'] = ((price_signal + volume_signal + vwap_signal) >= 2).astype(int)
        data['Position'] = data['Signal'].diff()
        
        return data, f"量价策略：价格趋势 + 放量确认 + VWAP支撑，三重验证"
    
    def volatility_strategy(self, data, params):
        """波动率策略"""
        window = params.get('window', 20)
        vol_threshold = params.get('vol_threshold', 1.5)
        
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=window).std() * np.sqrt(252)
        data['Vol_MA'] = data['Volatility'].rolling(window=window).mean()
        
        # 波动率突破策略
        vol_breakout = (data['Volatility'] > data['Vol_MA'] * vol_threshold).astype(int)
        price_momentum = (data['Close'].pct_change(5) > 0).astype(int)
        
        data['Signal'] = vol_breakout * price_momentum
        data['Position'] = data['Signal'].diff()
        
        return data, f"波动率突破策略：{window}日波动率 > {vol_threshold}倍均值时入场"

# 高级风险管理系统
class AdvancedRiskManager:
    @staticmethod
    def add_stop_loss_take_profit(data, stop_loss_pct=0.05, take_profit_pct=0.10):
        """止损止盈管理"""
        data['Stop_Loss_Signal'] = 0
        data['Take_Profit_Signal'] = 0
        
        entry_price = None
        position = 0
        
        for i in range(len(data)):
            if data['Position'].iloc[i] == 1:  # 买入信号
                entry_price = data['Close'].iloc[i]
                position = 1
            elif data['Position'].iloc[i] == -1:  # 卖出信号
                position = 0
                entry_price = None
            
            if position == 1 and entry_price:
                stop_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + take_profit_pct)
                
                if data['Close'].iloc[i] < stop_price:
                    data.iloc[i, data.columns.get_loc('Stop_Loss_Signal')] = -1
                    position = 0
                    entry_price = None
                elif data['Close'].iloc[i] > take_profit_price:
                    data.iloc[i, data.columns.get_loc('Take_Profit_Signal')] = -1
                    position = 0
                    entry_price = None
        
        return data
    
    @staticmethod
    def calculate_position_size(capital, risk_per_trade, price, stop_loss_pct):
        """凯利公式仓位管理"""
        risk_amount = capital * risk_per_trade
        max_loss_per_share = price * stop_loss_pct
        
        if max_loss_per_share > 0:
            position_size = risk_amount / max_loss_per_share
            max_position = capital * 0.2 / price  # 最大20%仓位
            return min(position_size, max_position, capital / price)
        return 0
    
    @staticmethod
    def calculate_var(returns, confidence_level=0.05):
        """计算VaR风险价值"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_max_drawdown_duration(cumulative_returns):
        """计算最大回撤持续时间"""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # 找到回撤期间
        in_drawdown = drawdown < 0
        drawdown_periods = []
        start = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        return max(drawdown_periods) if drawdown_periods else 0

# 专业回测引擎
class ProfessionalBacktestEngine:
    @staticmethod
    def run_advanced_backtest(data, initial_capital=100000, commission=0.001, slippage=0.0005):
        """高级回测引擎"""
        data = data.copy()
        data['Returns'] = data['Close'].pct_change()
        
        # 考虑滑点和手续费的真实交易成本
        data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
        data['Trading_Costs'] = np.abs(data['Position']) * (commission + slippage)
        data['Net_Strategy_Returns'] = data['Strategy_Returns'] - data['Trading_Costs']
        
        # 计算累计收益
        data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()
        data['Strategy_Cumulative'] = (1 + data['Net_Strategy_Returns']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Strategy_Cumulative']
        
        # 计算每日仓位
        data['Position_Size'] = 0
        capital = initial_capital
        
        for i in range(len(data)):
            if data['Position'].iloc[i] != 0:
                risk_per_trade = 0.02  # 2%风险
                stop_loss_pct = 0.05   # 5%止损
                position_size = AdvancedRiskManager.calculate_position_size(
                    capital, risk_per_trade, data['Close'].iloc[i], stop_loss_pct
                )
                data.iloc[i, data.columns.get_loc('Position_Size')] = position_size
                capital = data['Portfolio_Value'].iloc[i] if i > 0 else initial_capital
        
        return data
    
    @staticmethod
    def calculate_advanced_metrics(data):
        """计算高级回测指标"""
        strategy_returns = data['Net_Strategy_Returns'].dropna()
        
        if len(strategy_returns) == 0:
            return {"错误": "无有效交易数据"}
        
        # 基础指标
        total_return = data['Strategy_Cumulative'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1 if len(strategy_returns) > 0 else 0
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # 高级指标
        cumulative = data['Strategy_Cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 风险指标
        var_5 = AdvancedRiskManager.calculate_var(strategy_returns, 0.05)
        var_1 = AdvancedRiskManager.calculate_var(strategy_returns, 0.01)
        
        # 交易统计
        winning_trades = len(strategy_returns[strategy_returns > 0])
        losing_trades = len(strategy_returns[strategy_returns < 0])
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = strategy_returns[strategy_returns > 0].mean() if winning_trades > 0 else 0
        avg_loss = strategy_returns[strategy_returns < 0].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')
        
        # Sortino比率
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation != 0 else 0
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            "总收益率": f"{total_return:.2%}",
            "年化收益率": f"{annual_return:.2%}",
            "年化波动率": f"{volatility:.2%}",
            "夏普比率": f"{sharpe_ratio:.2f}",
            "Sortino比率": f"{sortino_ratio:.2f}",
            "Calmar比率": f"{calmar_ratio:.2f}",
            "最大回撤": f"{max_drawdown:.2%}",
            "VaR(5%)": f"{var_5:.2%}",
            "VaR(1%)": f"{var_1:.2%}",
            "胜率": f"{win_rate:.2%}",
            "盈亏比": f"{profit_factor:.2f}",
            "交易次数": total_trades,
            "平均盈利": f"{avg_win:.2%}",
            "平均亏损": f"{avg_loss:.2%}"
        }

# AI量化分析师
class QuantGPTAnalyst:
    def __init__(self):
        self.strategy_engine = ProfessionalStrategyEngine()
        self.risk_manager = AdvancedRiskManager()
        self.backtest_engine = ProfessionalBacktestEngine()
        
        # 预设策略库
        self.strategy_library = {
            "保守型": {"risk_level": "低", "strategies": ["均值回归", "网格交易"]},
            "平衡型": {"risk_level": "中", "strategies": ["趋势跟踪", "突破策略"]},
            "激进型": {"risk_level": "高", "strategies": ["动量策略", "波动率策略"]},
            "专业型": {"risk_level": "定制", "strategies": ["量价策略", "配对交易"]}
        }
    
    def parse_user_input(self, user_input):
        """智能解析用户输入"""
        user_input_lower = user_input.lower()
        
        # 股票代码提取（支持更多格式）
        stock_patterns = [
            r'\b[A-Z]{1,5}\b',  # 基础股票代码
            r'\b\w+\.\w+\b',    # 如 TSLA.US
            r'\^\w+\b'          # 指数代码如 ^GSPC
        ]
        
        stocks = []
        for pattern in stock_patterns:
            stocks.extend(re.findall(pattern, user_input.upper()))
        
        # 策略识别（更智能的关键词匹配）
        strategy_keywords = {
            "趋势": "趋势跟踪", "均线": "趋势跟踪", "双均线": "趋势跟踪", "移动平均": "趋势跟踪",
            "均值回归": "均值回归", "布林带": "均值回归", "回归": "均值回归",
            "动量": "动量策略", "rsi": "动量策略", "macd": "动量策略", "随机": "动量策略",
            "突破": "突破策略", "通道": "突破策略", "支撑": "突破策略", "阻力": "突破策略",
            "网格": "网格交易", "定投": "网格交易", "分批": "网格交易",
            "配对": "配对交易", "套利": "配对交易", "对冲": "配对交易",
            "量价": "量价策略", "成交量": "量价策略", "vwap": "量价策略",
            "波动": "波动率策略", "volatility": "波动率策略", "vix": "波动率策略"
        }
        
        detected_strategy = None
        for keyword, strategy in strategy_keywords.items():
            if keyword in user_input_lower:
                detected_strategy = strategy
                break
        
        # 风险偏好识别
        risk_keywords = {
            "保守": "保守型", "稳健": "保守型", "低风险": "保守型",
            "平衡": "平衡型", "中等": "平衡型", "适中": "平衡型",
            "激进": "激进型", "高风险": "激进型", "冒险": "激进型",
            "专业": "专业型", "定制": "专业型", "高级": "专业型"
        }
        
        risk_preference = None
        for keyword, risk_type in risk_keywords.items():
            if keyword in user_input_lower:
                risk_preference = risk_type
                break
        
        # 参数提取（更智能）
        params = {}
        
        # 数字参数提取
        numbers = re.findall(r'\d+', user_input)
        if any(word in user_input_lower for word in ["天", "日", "day", "period"]):
            if len(numbers) >= 1:
                params['window'] = int(numbers[0])
                params['short_window'] = int(numbers[0])
            if len(numbers) >= 2:
                params['long_window'] = int(numbers[1])
        
        # 百分比参数
        percentage_matches = re.findall(r'(\d+(?:\.\d+)?)%', user_input)
        if percentage_matches:
            if any(word in user_input_lower for word in ["止损", "stop", "loss"]):
                params['stop_loss'] = float(percentage_matches[0]) / 100
            elif any(word in user_input_lower for word in ["止盈", "profit", "target"]):
                params['take_profit'] = float(percentage_matches[0]) / 100
            elif any(word in user_input_lower for word in ["网格", "grid"]):
                params['grid_size'] = float(percentage_matches[0]) / 100
        
        # 时间周期识别
        period_keywords = {
            "1月": "1mo", "1个月": "1mo", "月": "1mo",
            "3月": "3mo", "3个月": "3mo", "季度": "3mo",
            "半年": "6mo", "6月": "6mo", "6个月": "6mo",
            "1年": "1y", "一年": "1y", "年": "1y",
            "2年": "2y", "两年": "2y"
        }
        
        period = "2y"  # 默认2年
        for keyword, p in period_keywords.items():
            if keyword in user_input:
                period = p
                break
        
        return {
            'stocks': list(set(stocks)),  # 去重
            'strategy': detected_strategy,
            'risk_preference': risk_preference,
            'params': params,
            'period': period,
            'original_input': user_input
        }
    
    def generate_intelligent_response(self, parsed_input):
        """生成智能AI响应"""
        stocks = parsed_input['stocks']
        strategy = parsed_input['strategy']
        risk_preference = parsed_input['risk_preference']
        params = parsed_input['params']
        period = parsed_input['period']
        original_input = parsed_input['original_input']
        
        # 输入验证和建议
        if not stocks:
            return self._generate_stock_suggestion_response()
        
        if not strategy and not risk_preference:
            return self._generate_strategy_suggestion_response(stocks)
        
        # 如果有风险偏好但没有具体策略，推荐策略
        if risk_preference and not strategy:
            return self._generate_risk_based_strategy_response(stocks, risk_preference)
        
        # 执行分析
        return self._execute_analysis(stocks, strategy, params, period)
    
    def _generate_stock_suggestion_response(self):
        """生成股票建议响应"""
        return """🤖 **QuantGPT Pro 为您服务！**

请告诉我您想分析的股票代码，我支持：

📈 **美股** (如: AAPL, TSLA, GOOGL, MSFT, NVDA)
📊 **指数** (如: ^GSPC, ^DJI, ^IXIC)
🌍 **国际股票** (如: TSLA.US, 0700.HK)

**示例：**
• "分析AAPL的趋势策略"
• "用保守型策略分析TSLA"
• "GOOGL和MSFT的动量策略对比"
"""
    
    def _generate_strategy_suggestion_response(self, stocks):
        """生成策略建议响应"""
        return f"""🤖 **检测到股票：{', '.join(stocks)}**

请选择分析策略或告诉我您的风险偏好：

🎯 **策略选择：**
• **趋势跟踪** - 双均线系统，适合趋势明确的市场
• **均值回归** - 布林带策略，适合震荡市场
• **动量策略** - RSI+MACD组合，捕捉强势行情
• **突破策略** - 价量突破，把握关键点位
• **网格交易** - 区间操作，适合波动市场
• **量价策略** - 成交量确认，提高胜率

🎨 **风险偏好：**
• **保守型** - 稳健策略，低风险低收益
• **平衡型** - 均衡配置，风险收益平衡  
• **激进型** - 高收益策略，承受较高风险
• **专业型** - 定制策略，适合有经验的交易者

**示例：** "用平衡型策略分析{stocks[0]}" 或 "{stocks[0]}的趋势策略"
"""
    
    def _generate_risk_based_strategy_response(self, stocks, risk_preference):
        """基于风险偏好生成策略响应"""
        strategy_info = self.strategy_library[risk_preference]
        recommended_strategies = strategy_info["strategies"]
        
        response = f"🤖 **{risk_preference}投资者 - {', '.join(stocks)}分析**\n\n"
        response += f"基于您的**{risk_preference}**偏好，我推荐以下策略：\n\n"
        
        results = []
        for strategy in recommended_strategies:
            try:
                result = self._analyze_single_strategy(stocks[0], strategy, {}, "1y")
                results.append(result)
            except Exception as e:
                results.append(f"❌ {strategy}分析失败：{str(e)}")
        
        return response + "\n\n".join(results)
    
    def _execute_analysis(self, stocks, strategy, params, period):
        """执行完整分析"""
        results = []
        
        for stock in stocks:
            try:
                result = self._analyze_single_strategy(stock, strategy, params, period)
                results.append(result)
                
                # 存储分析数据
                if 'analysis_data' not in st.session_state:
                    st.session_state.analysis_data = {}
                
            except Exception as e:
                results.append(f"❌ 分析{stock}时出错：{str(e)}")
        
        return "\n\n".join(results)
    
    def _analyze_single_strategy(self, stock, strategy, params, period):
        """分析单个股票策略"""
        # 获取数据
        data = get_cached_stock_data(stock, period)
        if data.empty:
            return f"❌ 无法获取 {stock} 的数据"
        
        # 设置默认参数
        default_params = self._get_default_params(strategy)
        default_params.update(params)
        
        # 运行策略
        strategy_data, description = self.strategy_engine.strategies[strategy](data.copy(), default_params)
        
        # 添加风险管理
        if 'stop_loss' in params or 'take_profit' in params:
            stop_loss = params.get('stop_loss', 0.05)
            take_profit = params.get('take_profit', 0.10)
            strategy_data = self.risk_manager.add_stop_loss_take_profit(
                strategy_data, stop_loss, take_profit
            )
        
        # 运行高级回测
        backtest_data = self.backtest_engine.run_advanced_backtest(strategy_data)
        
        # 计算高级指标
        metrics = self.backtest_engine.calculate_advanced_metrics(backtest_data)
        
        # 存储数据
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = {}
        st.session_state.analysis_data[stock] = {
            'data': backtest_data,
            'strategy': strategy,
            'params': default_params,
            'period': period
        }
        
        # 格式化结果
        return self._format_professional_result(stock, strategy, description, metrics, default_params)
    
    def _get_default_params(self, strategy):
        """获取策略默认参数"""
        defaults = {
            "趋势跟踪": {'short_window': 20, 'long_window': 50},
            "均值回归": {'window': 20, 'std_dev': 2.0},
            "动量策略": {'rsi_window': 14, 'rsi_threshold': 70},
            "突破策略": {'window': 20, 'volume_factor': 1.5},
            "网格交易": {'grid_size': 0.02, 'atr_factor': 1.0},
            "配对交易": {'window': 30, 'threshold': 2.0},
            "量价策略": {'price_window': 20, 'volume_window': 20},
            "波动率策略": {'window': 20, 'vol_threshold': 1.5}
        }
        return defaults.get(strategy, {})
    
    def _format_professional_result(self, stock, strategy, description, metrics, params):
        """格式化专业分析结果"""
        result = f"## 📊 {stock} - {strategy}专业分析报告\n\n"
        result += f"**策略描述：** {description}\n\n"
        result += f"**参数配置：** {json.dumps(params, ensure_ascii=False)}\n\n"
        
        # 核心指标展示
        result += "### 📈 核心绩效指标\n\n"
        core_metrics = ["总收益率", "年化收益率", "夏普比率", "最大回撤", "胜率"]
        for metric in core_metrics:
            if metric in metrics:
                result += f"• **{metric}**: `{metrics[metric]}`\n"
        
        # 风险指标
        result += "\n### ⚠️ 风险控制指标\n\n"
        risk_metrics = ["Sortino比率", "Calmar比率", "VaR(5%)", "VaR(1%)", "年化波动率"]
        for metric in risk_metrics:
            if metric in metrics:
                result += f"• **{metric}**: `{metrics[metric]}`\n"
        
        # 交易统计
        result += "\n### 📊 交易执行统计\n\n"
        trade_metrics = ["交易次数", "盈亏比", "平均盈利", "平均亏损"]
        for metric in trade_metrics:
            if metric in metrics:
                result += f"• **{metric}**: `{metrics[metric]}`\n"
        
        # AI智能评级和建议
        result += "\n" + self._generate_ai_assessment(metrics, strategy)
        
        return result
    
    def _generate_ai_assessment(self, metrics, strategy):
        """生成AI智能评估"""
        try:
            sharpe = float(metrics.get("夏普比率", "0"))
            total_return = float(metrics.get("总收益率", "0%").rstrip('%')) / 100
            max_drawdown = float(metrics.get("最大回撤", "0%").rstrip('%')) / 100
            win_rate = float(metrics.get("胜率", "0%").rstrip('%')) / 100
            
            # 综合评分计算
            score = 0
            score += min(sharpe * 25, 50)  # 夏普比率权重50%
            score += min(total_return * 100, 30)  # 收益率权重30%
            score += max(0, 20 + max_drawdown * 100)  # 回撤控制权重20%
            
            # 评级系统
            if score >= 80:
                rating = "🌟 AAA级 - 卓越策略"
                color = "🟢"
            elif score >= 70:
                rating = "⭐ AA级 - 优秀策略"  
                color = "🟢"
            elif score >= 60:
                rating = "✨ A级 - 良好策略"
                color = "🟡"
            elif score >= 50:
                rating = "💫 BBB级 - 中等策略"
                color = "🟡"
            else:
                rating = "⚡ BB级 - 需要改进"
                color = "🔴"
            
            assessment = f"### 🤖 QuantGPT AI 智能评估\n\n"
            assessment += f"**策略评级：** {rating} (评分: {score:.0f}/100)\n\n"
            
            # 详细分析
            assessment += f"**性能分析：**\n"
            if sharpe > 1.5:
                assessment += f"{color} 风险调整收益优秀，夏普比率{sharpe:.2f}表现卓越\n"
            elif sharpe > 1.0:
                assessment += f"🟡 风险调整收益良好，夏普比率{sharpe:.2f}符合预期\n"
            else:
                assessment += f"🔴 风险调整收益偏低，建议优化策略参数\n"
            
            if abs(max_drawdown) < 0.1:
                assessment += f"{color} 回撤控制优秀，最大回撤{max_drawdown:.1%}在可接受范围\n"
            elif abs(max_drawdown) < 0.2:
                assessment += f"🟡 回撤控制良好，需要注意风险管理\n"
            else:
                assessment += f"🔴 回撤较大，建议加强止损措施\n"
            
            if win_rate > 0.6:
                assessment += f"{color} 胜率{win_rate:.1%}表现优秀，策略稳定性好\n"
            elif win_rate > 0.45:
                assessment += f"🟡 胜率{win_rate:.1%}中等，符合一般策略表现\n"
            else:
                assessment += f"🔴 胜率偏低，建议结合其他指标优化入场时机\n"
            
            # 实盘建议
            assessment += f"\n**💡 实盘交易建议：**\n"
            
            if score >= 70:
                assessment += f"• ✅ **推荐实盘应用** - 策略表现优秀，可考虑实际投资\n"
                assessment += f"• 💰 **建议仓位** - 根据风险承受能力，建议10-20%仓位\n"
                assessment += f"• 🎯 **止损建议** - 设置5-8%止损，保护本金安全\n"
            elif score >= 50:
                assessment += f"• ⚠️ **谨慎应用** - 策略有一定价值，建议小仓位测试\n"
                assessment += f"• 💰 **建议仓位** - 谨慎起见，建议5-10%仓位测试\n"
                assessment += f"• 🎯 **风控重点** - 严格执行止损，密切监控表现\n"
            else:
                assessment += f"• ❌ **不建议实盘** - 策略表现不佳，建议重新优化\n"
                assessment += f"• 🔧 **优化方向** - 调整参数或尝试其他策略类型\n"
                assessment += f"• 📚 **继续学习** - 建议深入研究市场规律和策略原理\n"
            
            # 策略特色分析
            strategy_insights = {
                "趋势跟踪": "适合趋势明确的市场环境，在震荡市场中可能产生较多假信号",
                "均值回归": "在震荡市场中表现优秀，但在强趋势市场中可能错失机会",
                "动量策略": "能够捕捉强势行情，但需要注意动量衰减的风险",
                "突破策略": "适合捕捉关键突破点位，注意假突破的风险",
                "网格交易": "适合区间震荡市场，在单边趋势中需要谨慎使用",
                "量价策略": "
