require("dotenv").config();
const { RealAPIFallbackDemo } = require("./real-api-fallback-demo");

async function main() {
  console.log("正在启动MCP真实API降级演示...");
  
  // 从环境变量获取API密钥
  const claudeApiKey = process.env.CLAUDE_API_KEY || "";
  const deepSeekApiKey = process.env.DEEPSEEK_API_KEY || "";
  
  // 验证API密钥
  if (!claudeApiKey || !deepSeekApiKey) {
    console.error("错误: 缺少API密钥，请在.env文件中设置CLAUDE_API_KEY和DEEPSEEK_API_KEY");
    process.exit(1);
  }
  
  // 创建演示实例
  const demo = new RealAPIFallbackDemo(claudeApiKey, deepSeekApiKey, {
    maxRetries: 1,
    recoveryInterval: 60000  // 演示用1分钟间隔
  });
  
  // 运行演示
  await demo.runDemo();
}

// 处理错误
process.on("unhandledRejection", (error) => {
  console.error("未处理的Promise拒绝:", error);
  process.exit(1);
});

// 执行主函数
main().catch(error => {
  console.error("演示执行失败:", error);
  process.exit(1);
});
