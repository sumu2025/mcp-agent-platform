/**
 * 会话摘要生成器
 * 负责从会话内容生成结构化摘要
 */

class SummaryGenerator {
  /**
   * 生成会话摘要
   * @param {Array} messages 会话消息数组
   * @returns {Object} 摘要数据
   */
  static generateSummary(messages) {
    // 默认摘要结构
    const summary = {
      topics: [],
      decisions: [],
      progress: {},
      problems: [],
      nextSteps: [],
      summary: ''
    };
    
    try {
      // 提取完整会话文本
      const fullText = messages
        .map(msg => msg.content || '')
        .join('\n\n');
      
      // 识别讨论主题
      summary.topics = this.extractTopics(fullText);
      
      // 提取决策
      summary.decisions = this.extractDecisions(fullText);
      
      // 识别进度更新
      summary.progress = this.extractProgress(fullText);
      
      // 提取问题
      summary.problems = this.extractProblems(fullText);
      
      // 提取下一步计划
      summary.nextSteps = this.extractNextSteps(fullText);
      
      // 生成摘要文本
      summary.summary = this.createSummaryText(fullText, summary);
      
      return summary;
    } catch (error) {
      console.error('生成会话摘要失败:', error);
      summary.summary = '无法生成摘要。';
      return summary;
    }
  }
  
  /**
   * 提取讨论主题
   * @param {string} text 会话文本
   * @returns {Array} 主题数组
   */
  static extractTopics(text) {
    const topics = [];
    
    // 简单规则提取主题
    // 1. 寻找"讨论了..."或类似表述
    const discussionMatches = text.match(/讨论了([^。！？]+)[。！？]/g) || [];
    discussionMatches.forEach(match => {
      const topic = match.replace(/讨论了[：:]*\s*/, '').trim();
      if (topic && !topics.includes(topic)) {
        topics.push(topic);
      }
    });
    
    // 2. 寻找"关于..."的表述
    const aboutMatches = text.match(/关于([^，。！？]+)的[^，。！？]+/g) || [];
    aboutMatches.forEach(match => {
      const topic = match.replace(/关于/, '').replace(/的[^，。！？]+/, '').trim();
      if (topic && !topics.includes(topic)) {
        topics.push(topic);
      }
    });
    
    // 3. 寻找带有"开发"、"设计"、"实现"的句子
    const developMatches = text.match(/[开发|设计|实现]([^。！？]+)[。！？]/g) || [];
    developMatches.forEach(match => {
      const parts = match.split(/[开发|设计|实现]/);
      if (parts.length > 1) {
        const topic = parts[1].split(/[。！？]/)[0].trim();
        if (topic && !topics.includes(topic) && topic.length > 2) {
          topics.push(topic);
        }
      }
    });
    
    // 4. 分析关键技术术语
    const techTerms = this.extractTechnicalTerms(text);
    techTerms.forEach(term => {
      if (!topics.some(t => t.includes(term))) {
        const context = this.extractTermContext(text, term);
        if (context) {
          topics.push(context);
        }
      }
    });
    
    return topics.slice(0, 5); // 最多返回5个主题
  }
  
  /**
   * 提取技术术语
   * @param {string} text 会话文本
   * @returns {Array} 术语数组
   */
  static extractTechnicalTerms(text) {
    const terms = [];
    const techPatterns = [
      /\b(API|REST|GraphQL|JSON|XML|HTTP|WebSocket)\b/g,
      /\b(TypeScript|JavaScript|Python|HTML|CSS)\b/g,
      /\b(React|Vue|Angular|Node|Express)\b/g,
      /\b(Obsidian|Claude|DeepSeek|LLM|RAG|API)\b/g,
      /\b(数据库|索引|缓存|存储|向量)\b/g,
      /\b(模块|组件|接口|类|函数)\b/g
    ];
    
    techPatterns.forEach(pattern => {
      const matches = text.match(pattern) || [];
      matches.forEach(match => {
        if (!terms.includes(match)) {
          terms.push(match);
        }
      });
    });
    
    return terms;
  }
  
  /**
   * 提取术语上下文
   * @param {string} text 会话文本
   * @param {string} term 术语
   * @returns {string} 上下文
   */
  static extractTermContext(text, term) {
    const termIndex = text.indexOf(term);
    if (termIndex === -1) return null;
    
    // 提取术语所在的句子
    const sentenceStart = text.lastIndexOf('.', termIndex) + 1;
    const sentenceEnd = text.indexOf('.', termIndex + term.length);
    
    if (sentenceStart < 0 || sentenceEnd < 0) return term;
    
    const sentence = text.substring(sentenceStart, sentenceEnd).trim();
    if (sentence.length > 50) {
      // 如果句子太长，截取一部分
      const snippetStart = Math.max(0, termIndex - sentenceStart - 20);
      const snippetEnd = Math.min(sentence.length, termIndex - sentenceStart + term.length + 20);
      return `${term}(${sentence.substring(snippetStart, snippetEnd).trim()})`;
    }
    
    return sentence;
  }
  
  /**
   * 提取决策
   * @param {string} text 会话文本
   * @returns {Array} 决策数组
   */
  static extractDecisions(text) {
    const decisions = [];
    
    // 寻找决策相关表述
    const decisionPatterns = [
      /决定([^。！？]+)[。！？]/g,
      /确定([^。！？]+)[。！？]/g,
      /选择([^。！？]+)[。！？]/g,
      /采用([^。！？]+)[。！？]/g,
      /我们(应该|将会)([^。！？]+)[。！？]/g
    ];
    
    decisionPatterns.forEach(pattern => {
      const matches = text.match(pattern) || [];
      matches.forEach(match => {
        const decisionText = match.trim();
        
        // 提取决策主题（简单规则）
        let topic = '架构决策';
        if (decisionText.includes('优先级')) topic = '优先级';
        if (decisionText.includes('技术栈') || decisionText.includes('框架')) topic = '技术选择';
        if (decisionText.includes('实现方式')) topic = '实现方式';
        if (decisionText.includes('开发')) topic = '开发计划';
        if (decisionText.includes('API') || decisionText.includes('接口')) topic = 'API设计';
        
        // 避免重复
        if (!decisions.some(d => d.decision === decisionText)) {
          decisions.push({
            topic,
            decision: decisionText
          });
        }
      });
    });
    
    return decisions.slice(0, 5); // 最多返回5个决策
  }
  
  /**
   * 提取进度信息
   * @param {string} text 会话文本
   * @returns {Object} 进度数据
   */
  static extractProgress(text) {
    const progress = {
      achievements: [],
      components: []
    };
    
    // 寻找成就表述
    const achievementPatterns = [
      /完成了([^。！？]+)[。！？]/g,
      /实现了([^。！？]+)[。！？]/g,
      /开发了([^。！？]+)[。！？]/g,
      /已经([^。！？]+)好了/g,
      /成功([^。！？]+)[。！？]/g
    ];
    
    achievementPatterns.forEach(pattern => {
      const matches = text.match(pattern) || [];
      matches.forEach(match => {
        const achievement = match.trim();
        if (!progress.achievements.includes(achievement)) {
          progress.achievements.push(achievement);
        }
      });
    });
    
    // 提取组件状态
    const componentPatterns = [
      /([\w\d\u4e00-\u9fa5]+)组件(已经|还没有)([^，。！？]+)[，。！？]/g,
      /([\w\d\u4e00-\u9fa5]+)系统(已经|还没有)([^，。！？]+)[，。！？]/g,
      /([\w\d\u4e00-\u9fa5]+)功能(已经|还没有)([^，。！？]+)[，。！？]/g
    ];
    
    componentPatterns.forEach(pattern => {
      const matches = Array.from(text.matchAll(pattern)) || [];
      matches.forEach(match => {
        if (match.length >= 4) {
          const component = match[1];
          const status = match[2] === '已经' ? '完成' : '进行中';
          const details = match[3];
          
          progress.components.push({
            name: component,
            status,
            details: details.trim()
          });
        }
      });
    });
    
    // 提取进度百分比
    const percentMatches = text.match(/进度[^0-9]*([0-9]+)%/);
    if (percentMatches && percentMatches[1]) {
      progress.overallProgress = `${percentMatches[1]}%`;
    } else {
      // 尝试估算整体进度
      const estimateMatches = text.match(/(整体|总体|项目)[^。！？]*(完成|实现)[^。！？]*([0-9]+)[^。！？]*(成|半|部分)/);
      if (estimateMatches && estimateMatches[3]) {
        const percent = parseInt(estimateMatches[3]);
        if (percent > 0 && percent <= 100) {
          progress.overallProgress = `${percent}%`;
        }
      }
    }
    
    return progress;
  }
  
  /**
   * 提取问题
   * @param {string} text 会话文本
   * @returns {Array} 问题数组
   */
  static extractProblems(text) {
    const problems = [];
    
    // 寻找问题表述
    const problemPatterns = [
      /问题是([^。！？]+)[。！？]/g,
      /存在(着|的)问题([^。！？]+)[。！？]/g,
      /遇到([^。！？]+)的挑战/g,
      /面临的困难([^。！？]+)[。！？]/g,
      /需要解决([^。！？]+)[。！？]/g
    ];
    
    problemPatterns.forEach(pattern => {
      const matches = text.match(pattern) || [];
      matches.forEach(match => {
        const problemText = match.trim();
        if (!problems.includes(problemText)) {
          problems.push(problemText);
        }
      });
    });
    
    return problems.slice(0, 3); // 最多返回3个问题
  }
  
  /**
   * 提取下一步计划
   * @param {string} text 会话文本
   * @returns {Array} 计划数组
   */
  static extractNextSteps(text) {
    const nextSteps = [];
    
    // 寻找计划表述
    const stepPatterns = [
      /下一步([^。！？]+)[。！？]/g,
      /接下来([^。！？]+)[。！？]/g,
      /计划([^。！？]+)[。！？]/g,
      /需要([^。！？]+)[。！？]/g,
      /我们(应该|将会)([^。！？]+)[。！？]/g
    ];
    
    stepPatterns.forEach(pattern => {
      const matches = text.match(pattern) || [];
      matches.forEach(match => {
        if (match.includes('下一步') || match.includes('接下来') || 
            match.includes('计划') || match.includes('应该') || 
            match.includes('将会')) {
          
          const stepText = match.trim();
          if (!nextSteps.includes(stepText)) {
            nextSteps.push(stepText);
          }
        }
      });
    });
    
    return nextSteps.slice(0, 3); // 最多返回3个计划
  }
  
  /**
   * 创建摘要文本
   * @param {string} fullText 完整会话文本
   * @param {Object} extractedData 提取的数据
   * @returns {string} 摘要文本
   */
  static createSummaryText(fullText, extractedData) {
    // 生成结构化摘要
    let summary = '';
    
    // 添加主题
    if (extractedData.topics.length > 0) {
      summary += `讨论了以下主题：${extractedData.topics.join('、')}。\n\n`;
    }
    
    // 添加决策
    if (extractedData.decisions.length > 0) {
      summary += `做出了以下决策：\n`;
      extractedData.decisions.forEach(decision => {
        summary += `- ${decision.decision}\n`;
      });
      summary += '\n';
    }
    
    // 添加进度
    if (extractedData.progress.achievements && extractedData.progress.achievements.length > 0) {
      summary += `取得了以下进展：\n`;
      extractedData.progress.achievements.forEach(achievement => {
        summary += `- ${achievement}\n`;
      });
      summary += '\n';
    }
    
    // 添加组件状态
    if (extractedData.progress.components && extractedData.progress.components.length > 0) {
      summary += `组件状态更新：\n`;
      extractedData.progress.components.forEach(component => {
        summary += `- ${component.name}: ${component.status} (${component.details})\n`;
      });
      summary += '\n';
    }
    
    // 添加问题
    if (extractedData.problems.length > 0) {
      summary += `识别的问题：\n`;
      extractedData.problems.forEach(problem => {
        summary += `- ${problem}\n`;
      });
      summary += '\n';
    }
    
    // 添加下一步计划
    if (extractedData.nextSteps.length > 0) {
      summary += `下一步计划：\n`;
      extractedData.nextSteps.forEach(step => {
        summary += `- ${step}\n`;
      });
      summary += '\n';
    }
    
    if (extractedData.progress.overallProgress) {
      summary += `当前总体进度约为${extractedData.progress.overallProgress}。`;
    }
    
    return summary;
  }
}

module.exports = SummaryGenerator;
