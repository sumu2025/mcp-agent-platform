/**
 * Obsidian命令解析器
 * 将Claude生成的指令解析为具体的Obsidian操作
 */

class CommandParser {
  /**
   * 初始化命令解析器
   * @param {Object} obsidianInterface Obsidian接口实例
   */
  constructor(obsidianInterface) {
    this.obsidian = obsidianInterface;
    
    // 支持的命令类型
    this.commandTypes = {
      CREATE_NOTE: 'create_note',
      APPEND_NOTE: 'append_note',
      UPDATE_NOTE: 'update_note',
      READ_NOTE: 'read_note',
      DELETE_NOTE: 'delete_note',
      LIST_NOTES: 'list_notes',
      SEARCH_NOTES: 'search_notes',
      CREATE_FOLDER: 'create_folder',
      GET_INFO: 'get_info'
    };
  }
  
  /**
   * 解析命令字符串
   * @param {string} commandStr 命令字符串
   * @returns {Object} 解析后的命令对象
   */
  parseCommand(commandStr) {
    try {
      // 尝试作为JSON解析
      return this._parseJsonCommand(commandStr);
    } catch (jsonError) {
      // 如果不是有效的JSON，尝试解析标记语言
      try {
        return this._parseMarkupCommand(commandStr);
      } catch (markupError) {
        // 最后尝试解析纯文本命令
        try {
          return this._parseTextCommand(commandStr);
        } catch (textError) {
          throw new Error(`无法解析命令: ${commandStr}\n${jsonError}\n${markupError}\n${textError}`);
        }
      }
    }
  }
  
  /**
   * 解析JSON格式的命令
   * @param {string} commandStr 命令字符串
   * @returns {Object} 解析后的命令对象
   * @private
   */
  _parseJsonCommand(commandStr) {
    // 提取JSON部分
    let jsonStr = commandStr;
    
    // 尝试从代码块中提取JSON
    const jsonMatch = commandStr.match(/```(?:json)?\s*\n([\s\S]*?)\n```/);
    if (jsonMatch) {
      jsonStr = jsonMatch[1].trim();
    }
    
    // 解析JSON
    const command = JSON.parse(jsonStr);
    
    // 验证命令格式
    if (!command.type || !Object.values(this.commandTypes).includes(command.type)) {
      throw new Error(`未知的命令类型: ${command.type}`);
    }
    
    return command;
  }
  
  /**
   * 解析标记语言格式的命令
   * @param {string} commandStr 命令字符串
   * @returns {Object} 解析后的命令对象
   * @private
   */
  _parseMarkupCommand(commandStr) {
    // 查找命令标记
    const typeMatch = commandStr.match(/<command[^>]*type=["']([^"']+)["'][^>]*>/i);
    if (!typeMatch) {
      throw new Error('未找到命令类型标记');
    }
    
    const type = typeMatch[1].toLowerCase();
    
    // 验证命令类型
    if (!Object.values(this.commandTypes).includes(type)) {
      throw new Error(`未知的命令类型: ${type}`);
    }
    
    // 提取参数
    const params = {};
    
    // 提取路径
    const pathMatch = commandStr.match(/<path>(.*?)<\/path>/s);
    if (pathMatch) {
      params.path = pathMatch[1].trim();
    }
    
    // 提取内容
    const contentMatch = commandStr.match(/<content>([\s\S]*?)<\/content>/s);
    if (contentMatch) {
      params.content = contentMatch[1];
    }
    
    // 提取查询
    const queryMatch = commandStr.match(/<query>(.*?)<\/query>/s);
    if (queryMatch) {
      params.query = queryMatch[1].trim();
    }
    
    // 提取选项
    const optionsMatch = commandStr.match(/<options>([\s\S]*?)<\/options>/s);
    if (optionsMatch) {
      try {
        params.options = JSON.parse(optionsMatch[1]);
      } catch (e) {
        // 如果不是有效的JSON，尝试解析键值对
        const options = {};
        const optionLines = optionsMatch[1].split('\n');
        
        for (const line of optionLines) {
          const keyValue = line.split(':');
          if (keyValue.length === 2) {
            const key = keyValue[0].trim();
            let value = keyValue[1].trim();
            
            // 简单的类型转换
            if (value === 'true') value = true;
            else if (value === 'false') value = false;
            else if (!isNaN(Number(value))) value = Number(value);
            
            options[key] = value;
          }
        }
        
        params.options = options;
      }
    }
    
    return {
      type,
      ...params
    };
  }
  
  /**
   * 解析纯文本格式的命令
   * @param {string} commandStr 命令字符串
   * @returns {Object} 解析后的命令对象
   * @private
   */
  _parseTextCommand(commandStr) {
    // 根据关键词识别命令类型
    let type = null;
    const params = {};
    
    // 创建笔记
    if (commandStr.match(/创建笔记|新建笔记|create\s+note/i)) {
      type = this.commandTypes.CREATE_NOTE;
      
      // 提取路径
      const pathMatch = commandStr.match(/(?:路径|path)[：:]\s*([^\n]+)/i);
      if (pathMatch) {
        params.path = pathMatch[1].trim();
      }
      
      // 提取内容
      const contentMatch = commandStr.match(/(?:内容|content)[：:]\s*\n([\s\S]+)$/i);
      if (contentMatch) {
        params.content = contentMatch[1].trim();
      }
    }
    // 附加到笔记
    else if (commandStr.match(/附加|追加|append/i)) {
      type = this.commandTypes.APPEND_NOTE;
      
      // 提取路径
      const pathMatch = commandStr.match(/(?:路径|path)[：:]\s*([^\n]+)/i);
      if (pathMatch) {
        params.path = pathMatch[1].trim();
      }
      
      // 提取内容
      const contentMatch = commandStr.match(/(?:内容|content)[：:]\s*\n([\s\S]+)$/i);
      if (contentMatch) {
        params.content = contentMatch[1].trim();
      }
    }
    // 更新笔记
    else if (commandStr.match(/更新笔记|修改笔记|update\s+note/i)) {
      type = this.commandTypes.UPDATE_NOTE;
      
      // 提取路径
      const pathMatch = commandStr.match(/(?:路径|path)[：:]\s*([^\n]+)/i);
      if (pathMatch) {
        params.path = pathMatch[1].trim();
      }
      
      // 提取内容
      const contentMatch = commandStr.match(/(?:内容|content)[：:]\s*\n([\s\S]+)$/i);
      if (contentMatch) {
        params.content = contentMatch[1].trim();
      }
    }
    // 读取笔记
    else if (commandStr.match(/读取笔记|获取笔记|read\s+note/i)) {
      type = this.commandTypes.READ_NOTE;
      
      // 提取路径
      const pathMatch = commandStr.match(/(?:路径|path)[：:]\s*([^\n]+)/i);
      if (pathMatch) {
        params.path = pathMatch[1].trim();
      }
    }
    // 删除笔记
    else if (commandStr.match(/删除笔记|delete\s+note/i)) {
      type = this.commandTypes.DELETE_NOTE;
      
      // 提取路径
      const pathMatch = commandStr.match(/(?:路径|path)[：:]\s*([^\n]+)/i);
      if (pathMatch) {
        params.path = pathMatch[1].trim();
      }
    }
    // 列出笔记
    else if (commandStr.match(/列出笔记|获取笔记列表|list\s+notes/i)) {
      type = this.commandTypes.LIST_NOTES;
      
      // 提取文件夹
      const folderMatch = commandStr.match(/(?:文件夹|folder)[：:]\s*([^\n]+)/i);
      if (folderMatch) {
        params.path = folderMatch[1].trim();
      }
      
      // 提取选项
      params.options = {
        recursive: commandStr.includes('递归') || commandStr.includes('recursive')
      };
    }
    // 搜索笔记
    else if (commandStr.match(/搜索笔记|search\s+notes/i)) {
      type = this.commandTypes.SEARCH_NOTES;
      
      // 提取查询
      const queryMatch = commandStr.match(/(?:查询|query)[：:]\s*([^\n]+)/i);
      if (queryMatch) {
        params.query = queryMatch[1].trim();
      }
    }
    // 创建文件夹
    else if (commandStr.match(/创建文件夹|新建文件夹|create\s+folder/i)) {
      type = this.commandTypes.CREATE_FOLDER;
      
      // 提取路径
      const pathMatch = commandStr.match(/(?:路径|path)[：:]\s*([^\n]+)/i);
      if (pathMatch) {
        params.path = pathMatch[1].trim();
      }
    }
    // 获取库信息
    else if (commandStr.match(/获取库信息|库信息|vault\s+info/i)) {
      type = this.commandTypes.GET_INFO;
    }
    
    if (!type) {
      throw new Error('无法识别命令类型');
    }
    
    return {
      type,
      ...params
    };
  }
  
  /**
   * 执行命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   */
  async executeCommand(command) {
    try {
      const { type } = command;
      
      switch (type) {
        case this.commandTypes.CREATE_NOTE:
          return this._executeCreateNote(command);
          
        case this.commandTypes.APPEND_NOTE:
          return this._executeAppendNote(command);
          
        case this.commandTypes.UPDATE_NOTE:
          return this._executeUpdateNote(command);
          
        case this.commandTypes.READ_NOTE:
          return this._executeReadNote(command);
          
        case this.commandTypes.DELETE_NOTE:
          return this._executeDeleteNote(command);
          
        case this.commandTypes.LIST_NOTES:
          return this._executeListNotes(command);
          
        case this.commandTypes.SEARCH_NOTES:
          return this._executeSearchNotes(command);
          
        case this.commandTypes.CREATE_FOLDER:
          return this._executeCreateFolder(command);
          
        case this.commandTypes.GET_INFO:
          return this._executeGetInfo(command);
          
        default:
          throw new Error(`未知的命令类型: ${type}`);
      }
    } catch (error) {
      console.error('执行命令失败:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  /**
   * 执行创建笔记命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   * @private
   */
  async _executeCreateNote(command) {
    if (!command.path) {
      throw new Error('创建笔记命令缺少路径参数');
    }
    
    if (!command.content) {
      throw new Error('创建笔记命令缺少内容参数');
    }
    
    const success = await this.obsidian.writeNote(command.path, command.content);
    
    return {
      success,
      path: command.path
    };
  }
  
  /**
   * 执行附加到笔记命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   * @private
   */
  async _executeAppendNote(command) {
    if (!command.path) {
      throw new Error('附加到笔记命令缺少路径参数');
    }
    
    if (!command.content) {
      throw new Error('附加到笔记命令缺少内容参数');
    }
    
    // 读取现有内容
    let existingContent = '';
    try {
      existingContent = await this.obsidian.getNote(command.path);
    } catch (error) {
      // 笔记不存在，创建新笔记
      existingContent = '';
    }
    
    // 附加内容
    const newContent = `${existingContent}\n\n${command.content}`;
    
    // 写入笔记
    const success = await this.obsidian.writeNote(command.path, newContent);
    
    return {
      success,
      path: command.path
    };
  }
  
  /**
   * 执行更新笔记命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   * @private
   */
  async _executeUpdateNote(command) {
    if (!command.path) {
      throw new Error('更新笔记命令缺少路径参数');
    }
    
    if (!command.content) {
      throw new Error('更新笔记命令缺少内容参数');
    }
    
    const success = await this.obsidian.writeNote(command.path, command.content);
    
    return {
      success,
      path: command.path
    };
  }
  
  /**
   * 执行读取笔记命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   * @private
   */
  async _executeReadNote(command) {
    if (!command.path) {
      throw new Error('读取笔记命令缺少路径参数');
    }
    
    const content = await this.obsidian.getNote(command.path);
    const metadata = await this.obsidian.getNoteMetadata(command.path);
    
    return {
      success: true,
      path: command.path,
      content,
      metadata
    };
  }
  
  /**
   * 执行删除笔记命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   * @private
   */
  async _executeDeleteNote(command) {
    if (!command.path) {
      throw new Error('删除笔记命令缺少路径参数');
    }
    
    const success = await this.obsidian.deleteNote(command.path);
    
    return {
      success,
      path: command.path
    };
  }
  
  /**
   * 执行列出笔记命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   * @private
   */
  async _executeListNotes(command) {
    const folder = command.path || '';
    const options = command.options || { recursive: false };
    
    const notes = await this.obsidian.listNotes(folder, options);
    
    return {
      success: true,
      folder,
      count: notes.length,
      notes
    };
  }
  
  /**
   * 执行搜索笔记命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   * @private
   */
  async _executeSearchNotes(command) {
    if (!command.query) {
      throw new Error('搜索笔记命令缺少查询参数');
    }
    
    const options = command.options || { limit: 10 };
    
    const notes = await this.obsidian.searchNotes(command.query, options);
    
    return {
      success: true,
      query: command.query,
      count: notes.length,
      notes
    };
  }
  
  /**
   * 执行创建文件夹命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   * @private
   */
  async _executeCreateFolder(command) {
    if (!command.path) {
      throw new Error('创建文件夹命令缺少路径参数');
    }
    
    const success = await this.obsidian.createFolder(command.path);
    
    return {
      success,
      path: command.path
    };
  }
  
  /**
   * 执行获取库信息命令
   * @param {Object} command 命令对象
   * @returns {Promise<Object>} 执行结果
   * @private
   */
  async _executeGetInfo(command) {
    const info = await this.obsidian.getVaultInfo();
    
    return {
      success: true,
      info
    };
  }
}

module.exports = CommandParser;
