/**
 * DeepSeek客户端测试脚本
 * 用于验证DeepSeek API集成
 */

const { DeepSeekClient, utils } = require('../mcp/adapters/deepseek');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// 配置文件路径
const CONFIG_PATH = path.join(
  process.env.HOME || process.env.USERPROFILE, 
  '.mcp', 
  'config.json'
);

// 测试提示词
const TEST_PROMPTS = [
  "为一个文件监控系统写一个简单的事件处理器",
  "实现一个简单的LRU缓存，用于DeepSeek响应的缓存",
  "写一个函数来解析Markdown文件中的YAML前置元数据"
];

// 测试适配函数
async function testPromptAdaptation() {
  console.log('\n===== 测试提示词适配 =====\n');
  
  const claudePrompt = `<answer>下面是一个TypeScript的实现:</answer>

\`\`\`typescript
function parseMarkdown(content: string): { metadata: any; body: string } {
  // 实现解析逻辑
}
\`\`\``;

  console.log('原始Claude提示词:');
  console.log(claudePrompt);
  
  console.log('\n适配后提示词:');
  console.log(utils.adaptPrompt(claudePrompt));
}

// 测试生成函数
async function testGeneration(client) {
  console.log('\n===== 测试文本生成 =====\n');
  
  for (const prompt of TEST_PROMPTS) {
    console.log(`提示词: "${prompt}"`);
    console.log('生成中...');
    
    try {
      const response = await client.generateText(prompt, { max_tokens: 200 });
      console.log('\n生成结果:');
      console.log('----------------------------');
      console.log(response.slice(0, 200) + '...');
      console.log('----------------------------\n');
    } catch (error) {
      console.error('生成失败:', error.message);
    }
    
    // 在每个请求之间暂停以避免速率限制
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

// 测试结构化输出
async function testStructuredOutput(client) {
  console.log('\n===== 测试结构化输出 =====\n');
  
  const prompt = "创建一个JSON对象，包含以下信息：文件名、大小、创建日期和最后修改日期";
  
  try {
    console.log('生成结构化数据中...');
    const result = await client.generateStructured(prompt);
    console.log('\n结构化输出:');
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error('结构化输出失败:', error.message);
  }
}

// 主测试函数
async function runTests() {
  console.log('===== DeepSeek客户端测试 =====');
  
  // 加载配置
  let config;
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const configData = fs.readFileSync(CONFIG_PATH, 'utf8');
      config = JSON.parse(configData);
    } else {
      config = { deepseek: {} };
    }
  } catch (error) {
    console.warn('无法加载配置:', error.message);
    config = { deepseek: {} };
  }
  
  // 检查API密钥
  if (!config.deepseek || !config.deepseek.apiKey) {
    config.deepseek = config.deepseek || {};
    config.deepseek.apiKey = await promptForApiKey();
    saveConfig(config);
  }
  
  // 创建客户端
  const client = new DeepSeekClient({
    apiKey: config.deepseek.apiKey,
    model: config.deepseek.model || 'deepseek-chat'
  });
  
  // 测试API可用性
  console.log('\n检查DeepSeek API可用性...');
  const isAvailable = await client.checkAvailability();
  if (!isAvailable) {
    console.error('无法连接到DeepSeek API，请检查API密钥和网络连接。');
    process.exit(1);
  }
  console.log('DeepSeek API可用!\n');
  
  // 运行各种测试
  await testPromptAdaptation();
  await testGeneration(client);
  await testStructuredOutput(client);
  
  console.log('\n===== 测试完成 =====');
}

// 提示输入API密钥
async function promptForApiKey() {
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
      
      rl.close();
      resolve(apiKey);
    });
  });
}

// 保存配置
function saveConfig(config) {
  try {
    const configDir = path.dirname(CONFIG_PATH);
    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
    }
    
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
  } catch (error) {
    console.warn('无法保存配置:', error.message);
  }
}

// 运行测试
runTests().catch(console.error);
