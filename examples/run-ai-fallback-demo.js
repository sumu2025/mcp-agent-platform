require("dotenv").config();
const { AIModelFallbackDemo } = require("./ai-model-fallback-demo");

async function main() {
  const claudeApiKey = process.env.CLAUDE_API_KEY || "mock_claude_key";
  const deepSeekApiKey = process.env.DEEPSEEK_API_KEY || "mock_deepseek_key";
  
  console.log("==== MCP 智能模型降级演示 ====");
  console.log("本演示展示 Claude → DeepSeek 的无缝降级");
  
  const demo = new AIModelFallbackDemo(claudeApiKey, deepSeekApiKey);
  await demo.initialize();
  
  console.log("
1. 正常模式: 使用 Claude");
  const response1 = await demo.processPrompt("请描述一下神经网络的工作原理");
  console.log(`响应: ${response1}
`);
  
  console.log("2. 模拟 Claude 失败");
  demo.simulateClaudeFailure();
  
  console.log("
3. 降级模式: 自动切换到 DeepSeek");
  const response2 = await demo.processPrompt("继续解释反向传播算法");
  console.log(`响应: ${response2}
`);
  
  console.log("4. 重置模拟");
  demo.resetSimulation();
  
  console.log("
5. 恢复正常模式: 回到 Claude");
  const response3 = await demo.processPrompt("总结一下神经网络的优势");
  console.log(`响应: ${response3}
`);
  
  console.log("==== 演示完成 ====");
}

main().catch(error => {
  console.error("演示执行出错:", error);
  process.exit(1);
});

