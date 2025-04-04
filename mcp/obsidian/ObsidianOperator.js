/**
 * Obsidian操作接口
 * 提供对Obsidian操作的抽象，包括笔记创建、修改、删除等
 */

const ObsidianVault = require('./ObsidianVault');

/**
 * Obsidian操作类
 */
class ObsidianOperator {
  /**
   * 构造函数
   * @param {ObsidianVault} vault Obsidian文件库实例
   * @param {Object} options 配置选项
   */
  constructor(vault, options = {}) {
    if (!(vault instanceof ObsidianVault)) {
      throw new Error('vault参数必须是ObsidianVault实例');
    }
    
    this.vault = vault;
    this.options = {
      defaultTemplate: options.defaultTemplate || '',
      defaultLocation: options.defaultLocation || '',
      createIntermediateFolders: options.createIntermediateFolders !== false,
      ...options
    };
    
    this.operationHistory = [];
    this.operationQueue = [];
    this.isProcessing = false;
  }
  
  /**
   * 创建笔记
   * @param {string} title 笔记标题
   * @param {string} content 笔记内容
   * @param {Object} options 选项
   * @returns {Promise<Object>} 笔记对象
   */
  async createNote(title, content, options = {}) {
    // 构建笔记路径
    const location = options.location || this.options.defaultLocation;
    const path = this._buildNotePath(location, title);
    
    // 使用模板
    const template = options.template || this.options.defaultTemplate;
    let finalContent = content;
    
    if (template) {
      finalContent = await this._applyTemplate(template, {
        title,
        content,
        date: new Date(),
        ...options.templateVars
      });
    }
    
    // 创建笔记
    const note = await this.vault.createOrUpdateNote(path, finalContent, {
      createIfNotExists: true,
      ...options
    });
    
    // 记录操作
    this._recordOperation('create', path, { title, content: finalContent });
    
    return note;
  }
  
  /**
   * 更新笔记
   * @param {string} path 笔记路径
   * @param {string} content 笔记内容
   * @param {Object} options 选项
   * @returns {Promise<Object>} 笔记对象
   */
  async updateNote(path, content, options = {}) {
    // 获取原始笔记
    const originalNote = await this.vault.getNote(path).catch(() => null);
    
    // 更新笔记
    const note = await this.vault.createOrUpdateNote(path, content, options);
    
    // 记录操作
    this._recordOperation('update', path, { 
      content, 
      originalContent: originalNote?.content 
    });
    
    return note;
  }
  
  /**
   * 追加到笔记
   * @param {string} path 笔记路径
   * @param {string} content 要追加的内容
   * @param {Object} options 选项
   * @returns {Promise<Object>} 笔记对象
   */
  async appendToNote(path, content, options = {}) {
    // 获取原始笔记
    let originalNote;
    try {
      originalNote = await this.vault.getNote(path);
    } catch (error) {
      // 如果笔记不存在，且允许创建
      if (options.createIfNotExists) {
        return this.createNote(
          path.split('/').pop().replace(/\.md$/, ''), 
          content, 
          options
        );
      }
      throw error;
    }
    
    // 生成新内容
    const separator = options.separator || '\n\n';
    const newContent = originalNote.content + separator + content;
    
    // 更新笔记
    return this.updateNote(path, newContent, options);
  }
  
  /**
   * 在笔记中插入内容
   * @param {string} path 笔记路径
   * @param {string} content 要插入的内容
   * @param {Object} position 插入位置
   * @param {Object} options 选项
   * @returns {Promise<Object>} 笔记对象
   */
  async insertIntoNote(path, content, position, options = {}) {
    // 获取原始笔记
    const originalNote = await this.vault.getNote(path);
    const originalContent = originalNote.content;
    
    // 处理插入位置
    let insertPos = 0;
    
    if (typeof position === 'number') {
      // 直接使用数字位置
      insertPos = Math.max(0, Math.min(position, originalContent.length));
    } else if (position) {
      if (position.line !== undefined) {
        // 按行号插入
        const lines = originalContent.split('\n');
        const lineIndex = Math.max(0, Math.min(position.line, lines.length));
        
        // 计算行首位置
        let pos = 0;
        for (let i = 0; i < lineIndex; i++) {
          pos += lines[i].length + 1; // +1 for newline
        }
        
        insertPos = pos;
      } else if (position.marker) {
        // 按标记插入
        const markerPos = originalContent.indexOf(position.marker);
        if (markerPos >= 0) {
          insertPos = position.after ? 
            markerPos + position.marker.length : 
            markerPos;
        }
      }
    }
    
    // 生成新内容
    const newContent = 
      originalContent.substring(0, insertPos) + 
      content + 
      originalContent.substring(insertPos);
    
    // 更新笔记
    return this.updateNote(path, newContent, options);
  }
  
  /**
   * 删除笔记
   * @param {string} path 笔记路径
   * @returns {Promise<boolean>} 是否成功
   */
  async deleteNote(path) {
    // 获取原始笔记（用于撤销）
    const originalNote = await this.vault.getNote(path).catch(() => null);
    
    // 删除笔记
    const result = await this.vault.deleteNote(path);
    
    // 记录操作
    if (result && originalNote) {
      this._recordOperation('delete', path, { 
        originalContent: originalNote.content,
        originalMetadata: originalNote.metadata
      });
    }
    
    return result;
  }
  
  /**
   * 创建每日笔记
   * @param {Date} date 日期，默认今天
   * @param {Object} options 选项
   * @returns {Promise<Object>} 笔记对象
   */
  async createDailyNote(date = new Date(), options = {}) {
    // 格式化日期为文件名
    const dateStr = date.toISOString().split('T')[0]; // YYYY-MM-DD
    const title = options.titleFormat ? 
      this._formatDate(date, options.titleFormat) : 
      dateStr;
    
    // 创建笔记
    return this.createNote(title, '', {
      template: options.template || 'daily',
      location: options.location || 'Daily Notes',
      templateVars: { date },
      ...options
    });
  }
  
  /**
   * 将链接添加到笔记
   * @param {string} path 笔记路径
   * @param {string} targetPath 目标笔记路径
   * @param {string} linkText 链接文本，默认为目标笔记标题
   * @param {Object} options 选项
   * @returns {Promise<Object>} 笔记对象
   */
  async addLinkToNote(path, targetPath, linkText, options = {}) {
    // 获取目标笔记标题
    if (!linkText) {
      const targetNote = await this.vault.getNote(targetPath).catch(() => null);
      linkText = targetNote ? 
        targetNote.name : 
        targetPath.split('/').pop().replace(/\.md$/, '');
    }
    
    // 创建链接
    const link = `[[${targetPath}|${linkText}]]`;
    
    // 判断插入位置
    if (options.position) {
      return this.insertIntoNote(path, link, options.position, options);
    } else {
      return this.appendToNote(path, link, options);
    }
  }
  
  /**
   * 撤销上一次操作
   * @returns {Promise<boolean>} 是否成功
   */
  async undoLastOperation() {
    if (this.operationHistory.length === 0) {
      return false;
    }
    
    const lastOp = this.operationHistory.pop();
    
    try {
      switch (lastOp.type) {
        case 'create':
          // 撤销创建 = 删除
          await this.vault.deleteNote(lastOp.path);
          break;
          
        case 'update':
          // 撤销更新 = 恢复原内容
          if (lastOp.data.originalContent !== undefined) {
            await this.vault.createOrUpdateNote(
              lastOp.path, 
              lastOp.data.originalContent
            );
          }
          break;
          
        case 'delete':
          // 撤销删除 = 重新创建
          if (lastOp.data.originalContent !== undefined) {
            await this.vault.createOrUpdateNote(
              lastOp.path, 
              lastOp.data.originalContent, 
              { metadata: lastOp.data.originalMetadata }
            );
          }
          break;
          
        default:
          return false;
      }
      
      return true;
    } catch (error) {
      console.error('撤销操作失败:', error);
      // 恢复操作历史
      this.operationHistory.push(lastOp);
      return false;
    }
  }
  
  /**
   * 应用模板
   * @param {string} templateName 模板名称
   * @param {Object} vars 变量
   * @returns {Promise<string>} 处理后的内容
   * @private
   */
  async _applyTemplate(templateName, vars = {}) {
    // 在实际实现中，这里将加载并处理模板
    // 简单实现，将来可以替换为更复杂的模板系统
    let template = this.options.templates?.[templateName] || '';
    
    // 替换变量
    Object.entries(vars).forEach(([key, value]) => {
      const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
      template = template.replace(regex, value);
    });
    
    // 处理日期
    template = template.replace(/{{date:([^}]+)}}/g, (match, format) => {
      return this._formatDate(vars.date || new Date(), format);
    });
    
    return template;
  }
  
  /**
   * 格式化日期
   * @param {Date} date 日期
   * @param {string} format 格式
   * @returns {string} 格式化后的日期
   * @private
   */
  _formatDate(date, format) {
    // 简单日期格式化
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    return format
      .replace('YYYY', year)
      .replace('MM', month)
      .replace('DD', day)
      .replace('ddd', date.toLocaleDateString(undefined, { weekday: 'short' }))
      .replace('dddd', date.toLocaleDateString(undefined, { weekday: 'long' }));
  }
  
  /**
   * 构建笔记路径
   * @param {string} location 位置
   * @param {string} title 标题
   * @returns {string} 路径
   * @private
   */
  _buildNotePath(location, title) {
    // 清理标题，移除不允许的字符
    const cleanTitle = title.replace(/[\\/:*?"<>|]/g, '-');
    
    // 构建路径
    let path = cleanTitle;
    if (location) {
      path = `${location}/${cleanTitle}`;
    }
    
    // 确保有.md扩展名
    if (!path.endsWith('.md')) {
      path += '.md';
    }
    
    return path;
  }
  
  /**
   * 记录操作
   * @param {string} type 操作类型
   * @param {string} path 路径
   * @param {Object} data 相关数据
   * @private
   */
  _recordOperation(type, path, data = {}) {
    const operation = {
      type,
      path,
      data,
      timestamp: Date.now()
    };
    
    this.operationHistory.push(operation);
    
    // 限制历史记录长度
    if (this.operationHistory.length > 50) {
      this.operationHistory.shift();
    }
  }
}

module.exports = ObsidianOperator;
