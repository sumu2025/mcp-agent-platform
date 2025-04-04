#!/usr/bin/env node

/**
 * MCP代码生成器 - 命令行工具
 * 
 * 使用方式:
 * mcp-code-gen component "组件名称" --desc="描述" --type=class --out=./output.js
 * mcp-code-gen api "API名称" --desc="描述" --endpoints=./endpoints.json --out=./api.js
 * mcp-code-gen test "目标文件路径" --code=./source.js --framework=jest --out=./test.js
 * mcp-code-gen adapter "适配器名称" --source=System1 --target=System2 --sourceInterface=./source.json --targetInterface=./target.json --out=./adapter.js
 */

const fs = require('fs');
const path = require('path');
const { CodeGenerator } = require('../mcp/adapters/deepseek');
const prompts = require('../mcp/templates/prompts');
require('dotenv').config();

// 命令行参数解析
const args = process.argv.slice(2);
let command = args[0];
let name = args[1];
let options = {};

// 解析选项
for (let i = 2; i < args.length; i++) {
  const arg = args[i];
  if (arg.startsWith('--')) {
    const parts = arg.slice(2).split('=');
    if (parts.length === 2) {
      options[parts[0]] = parts[1];
    } else {
      options[parts[0]] = true;
    }
  }
}

// 创建代码生成器
function createGenerator() {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    console.error('错误: 未找到DEEPSEEK_API_KEY环境变量');
    console.error('请在.env文件中设置DEEPSEEK_API_KEY或通过命令行提供');
    process.exit(1);
  }

  return new CodeGenerator({
    apiKey,
    model: options.model || 'deepseek-coder',
    defaults: {
      temperature: parseFloat(options.temp || '0.2'),
      max_tokens: parseInt(options.maxTokens || '2000', 10),
      language: options.lang || 'typescript'
    }
  });
}

// 读取文件内容
function readFileContent(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf-8');
  } catch (err) {
    console.error(`错误: 无法读取文件 ${filePath}: ${err.message}`);
    process.exit(1);
  }
}

// 保存生成的代码
function saveCode(code, outputPath) {
  if (!outputPath) {
    // 如果没有指定输出路径，直接打印到控制台
    console.log('\n生成的代码:\n');
    console.log(code);
    return;
  }

  try {
    // 确保输出目录存在
    const dir = path.dirname(outputPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    // 写入文件
    fs.writeFileSync(outputPath, code);
    console.log(`成功! 代码已保存到: ${outputPath}`);
  } catch (err) {
    console.error(`错误: 无法保存到 ${outputPath}: ${err.message}`);
    process.exit(1);
  }
}

// 生成组件
async function generateComponent() {
  if (!name) {
    console.error('错误: 缺少组件名称');
    console.error('用法: mcp-code-gen component "组件名称" --desc="描述" --type=class --out=./output.js');
    process.exit(1);
  }

  console.log(`正在生成组件: ${name}...`);
  
  const generator = createGenerator();
  
  // 准备规格
  const spec = {
    name,
    description: options.desc || `${name} 组件`,
    type: options.type || 'class',
    interfaces: options.interfaces ? options.interfaces.split(',') : [],
    methods: options.methods ? options.methods.split(',') : []
  };
  
  try {
    // 生成代码
    const code = await generator.generateComponent(spec, {
      language: options.lang || 'typescript'
    });
    
    // 保存代码
    saveCode(code, options.out);
  } catch (err) {
    console.error(`生成失败: ${err.message}`);
    process.exit(1);
  }
}

// 生成API
async function generateApi() {
  if (!name) {
    console.error('错误: 缺少API名称');
    console.error('用法: mcp-code-gen api "API名称" --desc="描述" --endpoints=./endpoints.json --out=./api.js');
    process.exit(1);
  }

  console.log(`正在生成API: ${name}...`);
  
  const generator = createGenerator();
  
  // 读取端点信息
  let endpoints = [];
  if (options.endpoints) {
    if (fs.existsSync(options.endpoints)) {
      try {
        endpoints = JSON.parse(readFileContent(options.endpoints));
      } catch (err) {
        console.error(`错误: 端点文件解析失败: ${err.message}`);
        process.exit(1);
      }
    } else {
      console.error(`错误: 端点文件不存在: ${options.endpoints}`);
      process.exit(1);
    }
  }
  
  // 准备规格
  const spec = {
    name,
    description: options.desc || `${name} API客户端`,
    endpoints
  };
  
  try {
    // 生成代码
    const code = await generator.generateApi(spec, {
      language: options.lang || 'typescript'
    });
    
    // 保存代码
    saveCode(code, options.out);
  } catch (err) {
    console.error(`生成失败: ${err.message}`);
    process.exit(1);
  }
}

// 生成测试
async function generateTests() {
  if (!name) {
    console.error('错误: 缺少目标文件路径');
    console.error('用法: mcp-code-gen test "目标文件路径" --code=./source.js --framework=jest --out=./test.js');
    process.exit(1);
  }

  console.log(`正在生成测试: ${name}...`);
  
  const generator = createGenerator();
  
  // 读取源代码
  let targetCode = '';
  if (options.code) {
    targetCode = readFileContent(options.code);
  } else if (fs.existsSync(name)) {
    targetCode = readFileContent(name);
  } else {
    console.error(`错误: 未提供源代码且目标文件不存在: ${name}`);
    process.exit(1);
  }
  
  // 准备规格
  const spec = {
    targetFile: name,
    targetCode,
    testFramework: options.framework || 'jest'
  };
  
  try {
    // 生成代码
    const code = await generator.generateTests(spec, {
      language: options.lang || 'typescript'
    });
    
    // 保存代码
    saveCode(code, options.out);
  } catch (err) {
    console.error(`生成失败: ${err.message}`);
    process.exit(1);
  }
}

// 生成适配器
async function generateAdapter() {
  if (!name) {
    console.error('错误: 缺少适配器名称');
    console.error('用法: mcp-code-gen adapter "适配器名称" --source=System1 --target=System2 --sourceInterface=./source.json --targetInterface=./target.json --out=./adapter.js');
    process.exit(1);
  }

  if (!options.source || !options.target) {
    console.error('错误: 缺少源系统或目标系统名称');
    process.exit(1);
  }

  console.log(`正在生成适配器: ${name}...`);
  
  const generator = createGenerator();
  
  // 读取接口定义
  let sourceInterface = {};
  let targetInterface = {};
  
  if (options.sourceInterface) {
    try {
      sourceInterface = JSON.parse(readFileContent(options.sourceInterface));
    } catch (err) {
      console.error(`错误: 源接口文件解析失败: ${err.message}`);
      process.exit(1);
    }
  }
  
  if (options.targetInterface) {
    try {
      targetInterface = JSON.parse(readFileContent(options.targetInterface));
    } catch (err) {
      console.error(`错误: 目标接口文件解析失败: ${err.message}`);
      process.exit(1);
    }
  }
  
  // 准备规格
  const spec = {
    name,
    sourceName: options.source,
    targetName: options.target,
    sourceInterface,
    targetInterface
  };
  
  try {
    // 生成代码
    const code = await generator.generateAdapter(spec, {
      language: options.lang || 'typescript'
    });
    
    // 保存代码
    saveCode(code, options.out);
  } catch (err) {
    console.error(`生成失败: ${err.message}`);
    process.exit(1);
  }
}

// 显示帮助信息
function showHelp() {
  console.log(`
MCP代码生成器 - 使用DeepSeek API生成高质量代码

用法:
  mcp-code-gen <命令> <名称> [选项]

命令:
  component       生成组件代码
  api             生成API客户端代码
  test            生成测试代码
  adapter         生成适配器代码
  help            显示帮助信息

通用选项:
  --lang=<language>     编程语言 (默认: typescript)
  --out=<path>          输出文件路径 (默认: 输出到控制台)
  --temp=<value>        温度参数 (默认: 0.2)
  --model=<model>       DeepSeek模型 (默认: deepseek-coder)

组件命令选项:
  --desc=<text>         组件描述
  --type=<type>         组件类型 (class, function, module等)
  --interfaces=<list>   实现的接口列表 (逗号分隔)
  --methods=<list>      方法列表 (逗号分隔)

API命令选项:
  --desc=<text>         API描述
  --endpoints=<file>    端点定义JSON文件

测试命令选项:
  --code=<file>         源代码文件
  --framework=<name>    测试框架 (默认: jest)

适配器命令选项:
  --source=<name>       源系统名称
  --target=<name>       目标系统名称
  --sourceInterface=<file>  源接口定义JSON文件
  --targetInterface=<file>  目标接口定义JSON文件

示例:
  mcp-code-gen component "UserManager" --desc="用户管理组件" --type=class --out=./user-manager.ts
  mcp-code-gen api "DeepSeekAPI" --desc="DeepSeek API客户端" --endpoints=./endpoints.json
  mcp-code-gen test "./src/utils.js" --framework=jest --out=./tests/utils.test.js
  mcp-code-gen adapter "ModelAdapter" --source=Claude --target=DeepSeek --sourceInterface=./claude.json --targetInterface=./deepseek.json
  `);
}

// 主函数
async function main() {
  // 如果没有命令，显示帮助信息
  if (!command) {
    showHelp();
    process.exit(0);
  }

  // 执行命令
  switch (command.toLowerCase()) {
    case 'component':
      await generateComponent();
      break;
    case 'api':
      await generateApi();
      break;
    case 'test':
      await generateTests();
      break;
    case 'adapter':
      await generateAdapter();
      break;
    case 'help':
      showHelp();
      break;
    default:
      console.error(`错误: 未知命令: ${command}`);
      console.error('使用 mcp-code-gen help 获取帮助');
      process.exit(1);
  }
}

// 运行主函数
main().catch(err => {
  console.error(`发生错误: ${err.message}`);
  process.exit(1);
});
