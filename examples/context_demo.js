/**
 * 上下文管理演示
 * 展示如何加载项目上下文和生成会话摘要
 */

const { ContextManager, SummaryGenerator } = require('../mcp/core/context');

// 创建上下文管理器
const contextManager = new ContextManager();

// 演示会话
const demoConversation = [
  {
    role: 'user',
    content: '我们应该如何开始实现DeepSeek集成？'
  },
  {
    role: 'assistant',
    content: '我们可以从创建基本的API适配器开始，先实现简化版，以便在开发过程中使用。' +
      '这样可以缓解Claude API的token限制，并提高开发效率。\n\n' +
      '我建议先实现以下功能：\n' +
      '1. 基础的API客户端\n' +
      '2. 简单的提示词转换\n' +
      '3. 命令行开发辅助工具'
  },
  {
    role: 'user',
    content: '好的，我们决定先实现DeepSeek快速集成，然后再开始核心功能的开发。'
  },
  {
    role: 'assistant',
    content: '我已经完成了DeepSeek API客户端的基础实现，以及提示词转换工具。' +
      '这些组件已经可以用于开发辅助了。\n\n' +
      '另外，我还实现了上下文持久化系统，可以加载项目元数据和保存会话摘要，' +
      '这将大大提高我们的工作效率。\n\n' +
      '总体进度大约是10%，接下来我们应该开始MCP框架扩展的设计工作。'
  }
];

async function runDemo() {
  console.log('===== 上下文管理演示 =====\n');
  
  // 1. 加载项目上下文
  console.log('加载项目上下文...');
  const contextPrompt = contextManager.buildContextPrompt();
  console.log('\n' + contextPrompt);
  
  // 2. 生成会话摘要
  console.log('\n\n生成会话摘要...');
  const summary = SummaryGenerator.generateSummary(demoConversation);
  console.log('\n摘要主题:', summary.topics);
  console.log('摘要决策:', summary.decisions);
  console.log('进度更新:', summary.progress);
  console.log('\n摘要文本:\n', summary.summary);
  
  // 3. 保存会话摘要
  console.log('\n\n保存会话摘要...');
  contextManager.saveSessionSummary('demo-session', summary);
  console.log('摘要已保存!');
  
  // 4. 更新项目状态
  console.log('\n\n更新项目状态...');
  contextManager.updateMetadata({
    progress: {
      overallProgress: '10%',
      recentAchievements: [
        '实现DeepSeek API快速集成',
        '开发上下文持久化系统'
      ]
    },
    developmentPlan: {
      currentTasks: [
        {
          name: 'MCP框架扩展设计',
          status: '计划中',
          priority: '高'
        }
      ]
    }
  });
  console.log('项目状态已更新!');
  
  // 5. 重新加载上下文，展示变化
  console.log('\n\n重新加载上下文...');
  const updatedContext = contextManager.buildContextPrompt();
  console.log('\n' + updatedContext);
}

// 运行演示
runDemo().catch(console.error);
