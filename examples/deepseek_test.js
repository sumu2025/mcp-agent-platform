/**
 * DeepSeek API测试脚本
 * 
 * 测试DeepSeek API连接和基本功能
 * 使用方式: node examples/deepseek_test.js
 */

const { DeepSeekClient } = require('../mcp/adapters/deepseek');
const path = require('path');
const fs = require('fs');

// 配置文件路径
const CONFIG_PATH = path.join(
  process.env.HOME || process.env.USERPROFILE, 
  '.mcp', 
  'config.json'
);

async function testDeepSeek() {
  console.log('DeepSeek API测试脚本');
  console.log('--------------------------');
  
  // 加载配置
  let config;
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      config = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
      console.log('已加载配置文件');
    } else {
      console.error('找不到配置文件。请先运行 mcp-dev-assist --setup');
      process.exit(1);
    }
  } catch (error) {
    console.error('加载配置文件失败:', error.message);
    process.exit(1);
  }
  
  if (!config.deepseek || !config.deepseek.apiKey) {
    console.error('配置文件中缺少DeepSeek API密钥。请先运行 mcp-dev-assist --setup');
    process.exit(1);
  }
  
  // 创建客户端
  const client = new DeepSeekClient({
    apiKey: config.deepseek.apiKey,
    model: config.deepseek.model || 'deepseek-chat'
  });
  
  console.log('测试连接...');
  const isAvailable = await client.checkAvailability();
  if (!isAvailable) {
    console.error('无法连接到DeepSeek API。请检查API密钥和网络连接。');
    process.exit(1);
  }
  console.log('✓ 连接成功');
  
  // 测试基本文本生成
  console.log('\n测试基本文本生成...');
  try {
    const basicPrompt = "用一句话解释什么是Obsidian。";
    console.log(`提示: "${basicPrompt}"`);
    
    const response = await client.generateText(basicPrompt);
    console.log('响应:');
    console.log(response);
    console.log('✓ 基本文本生成成功');
  } catch (error) {
    console.error('基本文本生成失败:', error.message);
  }
  
  // 测试结构化输出
  console.log('\n测试结构化输出...');
  try {
    const structuredPrompt = "生成一个包含name和features字段的JSON对象，描述Obsidian软件。";
    console.log(`提示: "${structuredPrompt}"`);
    
    const response = await client.generateStructured(structuredPrompt);
    console.log('响应:');
    console.log(JSON.stringify(response, null, 2));
    console.log('✓ 结构化输出成功');
  } catch (error) {
    console.error('结构化输出失败:', error.message);
  }
  
  // 测试代码生成
  console.log('\n测试代码生成...');
  try {
    const codePrompt = "写一个TypeScript函数，用于解析Markdown文件的前置元数据。";
    console.log(`提示: "${codePrompt}"`);
    
    const response = await client.generateText(codePrompt);
    console.log('响应:');
    console.log(response);
    console.log('✓ 代码生成成功');
  } catch (error) {
    console.error('代码生成失败:', error.message);
  }
  
  console.log('\n所有测试完成');
}

// 运行测试
testDeepSeek().catch(console.error);
