/**
 * 会话上下文管理器
 * 负责管理项目元数据和会话状态，为AI助手提供上下文
 */

const fs = require('fs');
const path = require('path');

class ContextManager {
  /**
   * 初始化上下文管理器
   * @param {Object} options 配置选项
   */
  constructor(options = {}) {
    this.options = {
      metadataPath: options.metadataPath || path.join(__dirname, 'project_metadata.json'),
      summaryPath: options.summaryPath || path.join(__dirname, 'session_summaries'),
      maxSummaries: options.maxSummaries || 5,
      ...options
    };
    
    // 确保摘要目录存在
    if (!fs.existsSync(this.options.summaryPath)) {
      fs.mkdirSync(this.options.summaryPath, { recursive: true });
    }
    
    // 初始化项目元数据
    this.metadata = this.loadMetadata();
  }
  
  /**
   * 加载项目元数据
   * @returns {Object} 项目元数据
   */
  loadMetadata() {
    try {
      if (fs.existsSync(this.options.metadataPath)) {
        const data = fs.readFileSync(this.options.metadataPath, 'utf8');
        return JSON.parse(data);
      }
      console.warn('项目元数据文件不存在:', this.options.metadataPath);
      return {};
    } catch (error) {
      console.error('加载项目元数据失败:', error);
      return {};
    }
  }
  
  /**
   * 更新项目元数据
   * @param {Object} updates 更新内容
   */
  updateMetadata(updates) {
    try {
      // 深度合并更新
      this.metadata = this.deepMerge(this.metadata, updates);
      
      // 更新日期
      if (this.metadata.project) {
        this.metadata.project.updateDate = new Date().toISOString().split('T')[0];
      }
      
      // 保存到文件
      fs.writeFileSync(
        this.options.metadataPath, 
        JSON.stringify(this.metadata, null, 2)
      );
    } catch (error) {
      console.error('更新项目元数据失败:', error);
    }
  }
  
  /**
   * 保存会话摘要
   * @param {string} sessionId 会话ID
   * @param {Object} data 会话数据
   */
  saveSessionSummary(sessionId, data) {
    try {
      const timestamp = new Date().toISOString();
      const summary = {
        sessionId,
        timestamp,
        topics: data.topics || [],
        decisions: data.decisions || [],
        progress: data.progress || {},
        summary: data.summary || ''
      };
      
      // 保存摘要文件
      const summaryFile = path.join(
        this.options.summaryPath,
        `session_${sessionId}_${timestamp.replace(/[:.]/g, '-')}.json`
      );
      
      fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));
      
      // 更新项目进度
      if (data.progress || data.decisions) {
        this.updateProjectProgress(data);
      }
      
      // 管理摘要文件数量
      this.manageSessionSummaries();
    } catch (error) {
      console.error('保存会话摘要失败:', error);
    }
  }
  
  /**
   * 更新项目进度
   * @param {Object} data 会话数据
   */
  updateProjectProgress(data) {
    // 如果有决策记录，添加到元数据
    if (data.decisions && data.decisions.length > 0) {
      if (!this.metadata.context) this.metadata.context = {};
      if (!this.metadata.context.decisions) this.metadata.context.decisions = [];
      
      // 添加新决策
      data.decisions.forEach(decision => {
        this.metadata.context.decisions.push({
          date: new Date().toISOString().split('T')[0],
          ...decision
        });
      });
    }
    
    // 如果有进度更新
    if (data.progress) {
      if (!this.metadata.progress) this.metadata.progress = {};
      
      // 更新整体进度
      if (data.progress.overallProgress) {
        this.metadata.progress.overallProgress = data.progress.overallProgress;
      }
      
      // 更新最近成就
      if (data.progress.achievements && data.progress.achievements.length > 0) {
        if (!this.metadata.progress.recentAchievements) {
          this.metadata.progress.recentAchievements = [];
        }
        
        // 添加新成就并保持数组大小
        this.metadata.progress.recentAchievements = [
          ...data.progress.achievements,
          ...this.metadata.progress.recentAchievements
        ].slice(0, 5);
      }
      
      // 更新里程碑状态
      if (data.progress.milestones) {
        data.progress.milestones.forEach(milestone => {
          if (!this.metadata.progress.milestones) return;
          
          const existingIndex = this.metadata.progress.milestones.findIndex(
            m => m.name === milestone.name
          );
          
          if (existingIndex >= 0) {
            this.metadata.progress.milestones[existingIndex] = {
              ...this.metadata.progress.milestones[existingIndex],
              ...milestone
            };
          }
        });
      }
    }
    
    // 保存更新后的元数据
    this.updateMetadata({});
  }
  
  /**
   * 加载最近的会话摘要
   * @param {number} count 摘要数量
   * @returns {Array} 会话摘要数组
   */
  loadRecentSummaries(count = 3) {
    try {
      // 获取摘要文件列表
      const files = fs.readdirSync(this.options.summaryPath)
        .filter(file => file.startsWith('session_') && file.endsWith('.json'))
        .sort()
        .reverse()
        .slice(0, count);
      
      // 加载摘要内容
      return files.map(file => {
        const filePath = path.join(this.options.summaryPath, file);
        const content = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(content);
      });
    } catch (error) {
      console.error('加载会话摘要失败:', error);
      return [];
    }
  }
  
  /**
   * 管理会话摘要文件数量
   */
  manageSessionSummaries() {
    try {
      // 获取摘要文件列表
      const files = fs.readdirSync(this.options.summaryPath)
        .filter(file => file.startsWith('session_') && file.endsWith('.json'))
        .sort() // 按名称排序（包含时间戳）
        .reverse();
      
      // 如果超过最大数量，删除最旧的文件
      if (files.length > this.options.maxSummaries) {
        const filesToRemove = files.slice(this.options.maxSummaries);
        filesToRemove.forEach(file => {
          fs.unlinkSync(path.join(this.options.summaryPath, file));
        });
      }
    } catch (error) {
      console.error('管理会话摘要失败:', error);
    }
  }
  
  /**
   * 构建完整的上下文提示
   * @returns {string} 上下文提示
   */
  buildContextPrompt() {
    try {
      let prompt = '# MCP项目上下文\n\n';
      
      // 添加项目信息
      if (this.metadata.project) {
        prompt += `## 项目信息\n\n`;
        prompt += `- **项目名称**: ${this.metadata.project.name}\n`;
        prompt += `- **当前阶段**: ${this.metadata.project.stage}\n`;
        prompt += `- **版本**: ${this.metadata.project.version}\n`;
        prompt += `- **描述**: ${this.metadata.project.description}\n`;
        prompt += `- **最近更新**: ${this.metadata.project.updateDate}\n\n`;
      }
      
      // 添加架构信息
      if (this.metadata.architecture) {
        prompt += `## 项目架构\n\n`;
        prompt += `架构类型: ${this.metadata.architecture.type}\n`;
        prompt += `主要模式: ${this.metadata.architecture.primaryMode}\n`;
        
        if (this.metadata.architecture.fallbackModes) {
          prompt += `降级模式:\n`;
          this.metadata.architecture.fallbackModes.forEach(mode => {
            prompt += `- ${mode.name} (触发条件: ${mode.triggerCondition})\n`;
          });
        }
        
        prompt += '\n组件状态:\n';
        if (this.metadata.architecture.components) {
          this.metadata.architecture.components.forEach(component => {
            prompt += `- ${component.name}: ${component.status}\n`;
          });
        }
        prompt += '\n';
      }
      
      // 添加开发计划
      if (this.metadata.developmentPlan) {
        prompt += `## 当前开发状态\n\n`;
        prompt += `当前阶段: ${this.metadata.developmentPlan.currentPhase}\n`;
        prompt += `阶段时间: ${this.metadata.developmentPlan.startDate} 至 ${this.metadata.developmentPlan.endDate}\n\n`;
        
        prompt += `当前任务:\n`;
        if (this.metadata.developmentPlan.currentTasks) {
          this.metadata.developmentPlan.currentTasks.forEach(task => {
            prompt += `- ${task.name} (${task.status}, 优先级:${task.priority})\n`;
          });
        }
        prompt += '\n';
      }
      
      // 添加进度信息
      if (this.metadata.progress) {
        prompt += `## 项目进度\n\n`;
        prompt += `整体进度: ${this.metadata.progress.overallProgress}\n\n`;
        
        prompt += `最近成就:\n`;
        if (this.metadata.progress.recentAchievements) {
          this.metadata.progress.recentAchievements.forEach(achievement => {
            prompt += `- ${achievement}\n`;
          });
        }
        prompt += '\n';
      }
      
      // 添加核心决策
      if (this.metadata.context && this.metadata.context.decisions) {
        prompt += `## 核心决策\n\n`;
        
        // 只取最近的3个决策
        const recentDecisions = this.metadata.context.decisions
          .sort((a, b) => new Date(b.date) - new Date(a.date))
          .slice(0, 3);
        
        recentDecisions.forEach(decision => {
          prompt += `- **${decision.topic}** (${decision.date}): ${decision.decision}\n`;
          if (decision.rationale) {
            prompt += `  理由: ${decision.rationale}\n`;
          }
        });
        prompt += '\n';
      }
      
      // 添加最近会话摘要
      const recentSummaries = this.loadRecentSummaries();
      if (recentSummaries.length > 0) {
        prompt += `## 最近会话摘要\n\n`;
        
        recentSummaries.forEach(summary => {
          const date = new Date(summary.timestamp).toLocaleDateString();
          prompt += `### 会话 ${date}\n\n`;
          
          if (summary.topics && summary.topics.length > 0) {
            prompt += `讨论主题: ${summary.topics.join(', ')}\n`;
          }
          
          if (summary.summary) {
            prompt += `${summary.summary}\n`;
          }
          
          prompt += '\n';
        });
      }
      
      return prompt;
    } catch (error) {
      console.error('构建上下文提示失败:', error);
      return '无法加载项目上下文。';
    }
  }
  
  /**
   * 深度合并对象
   * @param {Object} target 目标对象
   * @param {Object} source 源对象
   * @returns {Object} 合并后的对象
   */
  deepMerge(target, source) {
    const output = { ...target };
    
    if (this.isObject(target) && this.isObject(source)) {
      Object.keys(source).forEach(key => {
        if (this.isObject(source[key])) {
          if (!(key in target)) {
            Object.assign(output, { [key]: source[key] });
          } else {
            output[key] = this.deepMerge(target[key], source[key]);
          }
        } else {
          Object.assign(output, { [key]: source[key] });
        }
      });
    }
    
    return output;
  }
  
  /**
   * 检查值是否为对象
   * @param {*} item 要检查的值
   * @returns {boolean} 是否为对象
   */
  isObject(item) {
    return item && typeof item === 'object' && !Array.isArray(item);
  }
}

module.exports = ContextManager;
