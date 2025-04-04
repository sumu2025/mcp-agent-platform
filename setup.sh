#!/bin/bash

# 设置脚本，为CLI工具添加执行权限并安装依赖

echo "设置MCP智能体中台开发环境..."

# 添加执行权限
echo "添加CLI工具执行权限..."
chmod +x ./bin/mcp-dev-assist.js

# 安装依赖
echo "安装依赖..."
npm install

# 创建配置目录
echo "创建配置目录..."
mkdir -p ~/.mcp

# 尝试创建符号链接
echo "创建全局命令链接..."
npm link || echo "无法创建全局链接，请使用sudo npm link或使用npm run dev-assist替代"

echo "设置完成！"
echo "使用方式: mcp-dev-assist \"您的开发问题\""
echo "查看详细说明请参考 DEEPSEEK_SETUP.md"
