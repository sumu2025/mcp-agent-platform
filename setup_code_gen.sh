#!/bin/bash

# 设置mcp-code-gen工具

echo "设置MCP代码生成器..."

# 添加执行权限
echo "添加执行权限..."
chmod +x ./bin/mcp-code-gen.js

# 安装依赖
echo "安装依赖..."
npm install dotenv

# 创建符号链接
echo "创建全局命令链接..."
npm link || echo "无法创建全局链接，请使用sudo npm link或使用npm run code-gen替代"

echo "设置完成！"
echo "使用方式: mcp-code-gen help"
