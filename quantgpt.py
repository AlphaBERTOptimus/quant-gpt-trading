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
        
        elif
