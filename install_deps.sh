#!/bin/bash
# 安装项目依赖脚本

# 更新pip
pip install --upgrade pip

# 安装基本依赖
pip install requests>=2.28.0 pydantic>=2.0.0 typer[all]>=0.9.0 python-dotenv>=1.0.0 rich>=13.0.0

# 安装开发依赖
pip install pytest>=7.0.0 black>=23.0.0 isort>=5.12.0

# 尝试重新安装项目
pip install -e .

echo "依赖安装完成，请运行测试脚本检查是否成功"
