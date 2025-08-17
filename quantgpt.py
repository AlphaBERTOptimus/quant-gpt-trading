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
