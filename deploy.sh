#!/bin/bash

# QuantGPT Pro 自动部署脚本
# 使用方法: chmod +x deploy.sh && ./deploy.sh

set -e

echo "🚀 QuantGPT Pro 自动部署脚本"
echo "================================"

# 检查Python版本
echo "📋 检查Python环境..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查是否存在虚拟环境
if [ ! -d "venv" ]; then
    echo "🔧 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "⚡ 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "📦 升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📚 安装项目依赖..."
pip install -r requirements.txt

# 检查Streamlit安装
echo "🔍 验证Streamlit安装..."
streamlit version

# 创建必要的目录
echo "📁 创建项目目录结构..."
mkdir -p .streamlit
mkdir -p assets/images
mkdir -p assets/data

# 检查配置文件
if [ ! -f ".streamlit/config.toml" ]; then
    echo "⚠️  警告: .streamlit/config.toml 文件不存在"
    echo "请确保配置文件已正确创建"
fi

# 运行测试
echo "🧪 运行基础测试..."
python3 -c "
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
print('✅ 所有核心依赖包导入成功')
"

echo ""
echo "✅ 部署准备完成！"
echo ""
echo "🎯 启动应用请运行:"
echo "   streamlit run app.py"
echo ""
echo "🌐 本地访问地址:"
echo "   http://localhost:8501"
echo ""
echo "📚 更多帮助:"
echo "   - GitHub: https://github.com/yourusername/quantgpt-pro"
echo "   - 文档: https://docs.streamlit.io"
echo ""
echo "🚀 Happy Trading!"
