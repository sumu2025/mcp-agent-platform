#!/usr/bin/env node

/**
 * MCP开发助手CLI工具
 * 使用DeepSeek API提供开发辅助
 * 
 * 使用方式:
 * mcp-dev-assist "为用户管理系统设计数据库模式"
 * mcp-dev-assist --file input.txt
 * mcp-dev-assist --code "function hello() { return 'world'; }"
 */

const path = require('path');
const fs = require('fs');
const readline = require('readline');
const { DeepSeekClient } = require('../mcp/adapters/deepseek');

// 配置文件路径
const CONFIG_DIR = path.join(
  process.env.HOME || process.env.USERPROFILE, 
  '.mcp'
);

const CONFIG_PATH = path.join(CONFIG_DIR, 'config.json');
const CACHE_DIR = path.join(CONFIG_DIR, 'cache');

// 确保目录存在
if (!fs.existsSync(CONFIG_DIR)) {
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
}

if (!fs.existsSync(CACHE_DIR)) {
  fs.mkdirSync(CACHE_DIR, { recursive: true });
}

// 助手类型前缀
const PREFIXES = {
  default: `你是MCP智能体中台的开发助手。帮助开发者编写代码、设计系统、解决问题。
给出简洁、实用、高质量的建议和实现。偏好TypeScript实现。
以下是开发者的请求：

`,
  code: `你是代码专家。请分析、改进或实现以下代码。给出简洁、高效、可维护的解决方案。
偏好TypeScript，遵循现代最佳实践。所有输出应当是可直接使用的完整代码。
以下是代码或代码请求：

`,
  design: `你是系统架构专家。帮助分析和设计软件系统、API和数据模型。
提供清晰、模块化、可扩展的设计方案，包括关键组件、接口定义和数据流。
以下是设计需求：

`,
  debug: `你是调试专家。帮助分析和解决以下代码或系统中的问题。
提供具体的错误诊断、根本原因分析和修复建议。
以下是需要调试的问题：

`
};

// 命令行参数解析
const args = process.argv.slice(2);
let mode = 'default';
let prompt = '';
let outputFile = null;

// 解析参数
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  
  if (arg === '--code' || arg === '-c') {
    mode = 'code';
    continue;
  }
  
  if (arg === '--design' || arg === '-d') {
    mode = 'design';
    continue;
  }
  
  if (arg === '--debug') {
    mode = 'debug';
    continue;
  }
  
  if (arg === '--file' || arg === '-f') {
    if (i + 1 < args.length) {
      const filePath = args[++i];
      try {
        prompt = fs.readFileSync(filePath, 'utf8');
      } catch (error) {
        console.error(`无法读取文件 ${filePath}: ${error.message}`);
        process.exit(1);
      }
    }
    continue;
  }
  
  if (arg === '--output' || arg === '-o') {
    if (i + 1 < args.length) {
      outputFile = args[++i];
    }
    continue;
  }
  
  if (arg === '--help' || arg === '-h') {
    printHelp();
    process.exit(0);
  }
  
  // 如果没有特殊标记，则作为提示内容
  if (!prompt) {
    prompt = arg;
  } else {
    prompt += ' ' + arg;
  }
}

// 主函数
async function main() {
  try {
    // 检查是否有足够的参数
    if (!prompt) {
      console.error('请提供开发问题或请求。例如：\nmcp-dev-assist "为用户管理系统设计数据库模式"');
      console.error('使用 --help 获取更多帮助。');
      process.exit(1);
    }

    // 加载配置
    const config = await loadConfig();
    
    if (!config.deepseek || !config.deepseek.apiKey) {
      await promptForApiKey(config);
    }

    // 创建DeepSeek客户端
    const client = new DeepSeekClient({
      apiKey: config.deepseek.apiKey,
      model: config.deepseek.model || 'deepseek-chat'
    });

    // 测试连接
    console.log('正在连接DeepSeek API...');
    const isAvailable = await client.checkAvailability();
    if (!isAvailable) {
      console.error('无法连接到DeepSeek API，请检查API密钥和网络连接。');
      process.exit(1);
    }

    // 检查缓存
    const cacheKey = getCacheKey(mode, prompt);
    const cachedResponse = checkCache(cacheKey);
    
    if (cachedResponse) {
      console.log('\n=== 缓存的回答 ===\n');
      console.log(cachedResponse);
      
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
      });
      
      const answer = await new Promise(resolve => {
        rl.question('\n使用缓存的回答？(Y/n) ', resolve);
      });
      
      rl.close();
      
      if (answer.toLowerCase() !== 'n') {
        saveToFileIfNeeded(cachedResponse, outputFile);
        return;
      }
      
      console.log('正在生成新的回答...');
    }

    // 准备完整提示
    const fullPrompt = PREFIXES[mode] + prompt;

    // 获取响应
    console.log('正在生成响应...');
    const response = await client.generateText(fullPrompt);
    
    // 缓存响应
    saveToCache(cacheKey, response);
    
    // 打印响应
    console.log('\n=== MCP开发助手回答 ===\n');
    console.log(response);
    console.log('\n=========================\n');
    
    // 保存到文件如果需要
    saveToFileIfNeeded(response, outputFile);

  } catch (error) {
    console.error('错误:', error.message);
    process.exit(1);
  }
}

/**
 * 保存到文件（如果指定了输出文件）
 */
function saveToFileIfNeeded(content, filePath) {
  if (filePath) {
    try {
      fs.writeFileSync(filePath, content);
      console.log(`回答已保存到 ${filePath}`);
    } catch (error) {
      console.error(`无法保存到文件 ${filePath}: ${error.message}`);
    }
  }
}

/**
 * 获取缓存键
 */
function getCacheKey(mode, prompt) {
  const hash = require('crypto')
    .createHash('md5')
    .update(`${mode}:${prompt}`)
    .digest('hex');
  
  return hash;
}

/**
 * 检查缓存
 */
function checkCache(cacheKey) {
  const cacheFile = path.join(CACHE_DIR, `${cacheKey}.txt`);
  
  if (fs.existsSync(cacheFile)) {
    try {
      const content = fs.readFileSync(cacheFile, 'utf8');
      return content;
    } catch (error) {
      console.warn(`读取缓存失败: ${error.message}`);
    }
  }
  
  return null;
}

/**
 * 保存到缓存
 */
function saveToCache(cacheKey, content) {
  const cacheFile = path.join(CACHE_DIR, `${cacheKey}.txt`);
  
  try {
    fs.writeFileSync(cacheFile, content);
  } catch (error) {
    console.warn(`保存缓存失败: ${error.message}`);
  }
}

/**
 * 加载配置文件
 * @returns {Promise<Object>} 配置对象
 */
async function loadConfig() {
  try {
    // 读取配置文件
    if (fs.existsSync(CONFIG_PATH)) {
      const configData = fs.readFileSync(CONFIG_PATH, 'utf8');
      return JSON.parse(configData);
    }

    // 返回默认配置
    return { deepseek: {} };
  } catch (error) {
    console.warn('无法加载配置:', error.message);
    return { deepseek: {} };
  }
}

/**
 * 保存配置文件
 * @param {Object} config 配置对象
 */
function saveConfig(config) {
  try {
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
  } catch (error) {
    console.warn('无法保存配置:', error.message);
  }
}

/**
 * 提示用户输入API密钥
 * @param {Object} config 配置对象
 */
async function promptForApiKey(config) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve, reject) => {
    rl.question('请输入DeepSeek API密钥: ', (apiKey) => {
      if (!apiKey) {
        rl.close();
        reject(new Error('API密钥不能为空'));
        return;
      }
      
      // 询问模型
      rl.question('使用哪个模型 (默认: deepseek-chat): ', (model) => {
        // 更新配置
        config.deepseek = config.deepseek || {};
        config.deepseek.apiKey = apiKey;
        config.deepseek.model = model || 'deepseek-chat';
        
        // 保存配置
        saveConfig(config);
        
        rl.close();
        resolve();
      });
    });
  });
}

/**
 * 打印帮助信息
 */
function printHelp() {
  console.log(`
MCP开发助手 - 使用DeepSeek API提供开发支持

用法:
  mcp-dev-assist [选项] [提示]
  mcp-dev-assist [选项] --file 文件路径

选项:
  --code, -c          代码模式 (代码分析、实现或改进)
  --design, -d        设计模式 (系统设计和架构)
  --debug             调试模式 (问题排查和解决)
  --file, -f 文件路径  从文件读取提示
  --output, -o 文件路径 将回答保存到文件
  --help, -h          显示帮助信息

示例:
  mcp-dev-assist "创建一个处理文件上传的API"
  mcp-dev-assist --code "实现一个带缓存的HTTP客户端"
  mcp-dev-assist --design "设计一个分布式任务队列系统"
  mcp-dev-assist --file request.txt --output response.md
  `);
}

// 运行主函数
main().catch(console.error);
