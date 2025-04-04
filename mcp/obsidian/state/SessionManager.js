/**
 * SessionManager
 * 
 * 管理与Obsidian交互的会话状态
 */

const EventEmitter = require('events');

/**
 * 会话状态
 */
class SessionState {
  /**
   * 初始化会话状态
   */
  constructor() {
    this.currentNote = null;       // 当前正在编辑的笔记
    this.selectedText = '';        // 当前选中的文本
    this.recentNotes = [];         // 最近访问的笔记
    this.context = {};             // 上下文信息
    this.history = [];             // 操作历史
    this.pending = [];             // 待执行操作
    this.timestamp = Date.now();   // 最后更新时间戳
  }
  
  /**
   * 更新会话状态
   * @param {Object} updates 状态更新
   */
  update(updates) {
    // 应用更新
    Object.entries(updates).forEach(([key, value]) => {
      if (key in this && key !== 'timestamp') {
        this[key] = value;
      }
    });
    
    // 更新时间戳
    this.timestamp = Date.now();
  }
  
  /**
   * 序列化状态为JSON
   * @returns {Object} 序列化后的状态
   */
  toJSON() {
    return {
      currentNote: this.currentNote,
      selectedText: this.selectedText,
      recentNotes: this.recentNotes.slice(0, 5), // 只保留最近5个
      context: this.context,
      timestamp: this.timestamp
    };
  }
}

/**
 * 会话管理器
 */
class SessionManager extends EventEmitter {
  /**
   * 初始化会话管理器
   * @param {Object} options 配置选项
   */
  constructor(options = {}) {
    super();
    
    this.options = {
      maxHistory: 20,          // 最大历史记录数
      historyExpiry: 3600000,  // 历史过期时间（毫秒）
      ...options
    };
    
    this.state = new SessionState();
    this.sessionId = this._generateSessionId();
  }
  
  /**
   * 更新会话状态
   * @param {Object} updates 状态更新
   */
  update(updates) {
    // 保存之前的状态
    const previousState = { ...this.state };
    
    // 更新状态
    this.state.update(updates);
    
    // 记录历史
    if (updates.currentNote) {
      this._addToHistory('note', updates.currentNote);
    }
    
    // 发出状态变更事件
    this.emit('state:changed', {
      previousState,
      currentState: this.state,
      changes: updates,
      timestamp: Date.now()
    });
  }
  
  /**
   * 获取当前状态
   * @returns {SessionState} 当前状态
   */
  getState() {
    return this.state;
  }
  
  /**
   * 获取会话ID
   * @returns {string} 会话ID
   */
  getSessionId() {
    return this.sessionId;
  }
  
  /**
   * 设置当前笔记
   * @param {Object} note 笔记对象
   */
  setCurrentNote(note) {
    if (!note) return;
    
    const noteInfo = {
      path: note.path,
      filename: note.filename,
      id: note.id
    };
    
    // 更新状态
    this.update({
      currentNote: noteInfo
    });
    
    // 更新最近笔记列表
    this._updateRecentNotes(noteInfo);
  }
  
  /**
   * 设置选中文本
   * @param {string} text 选中的文本
   */
  setSelectedText(text) {
    this.update({
      selectedText: text || ''
    });
  }
  
  /**
   * 添加上下文信息
   * @param {string} key 上下文键
   * @param {*} value 上下文值
   */
  addContext(key, value) {
    const contextUpdate = {
      ...this.state.context,
      [key]: value
    };
    
    this.update({
      context: contextUpdate
    });
  }
  
  /**
   * 清除上下文信息
   * @param {string} key 上下文键
   */
  clearContext(key) {
    const contextUpdate = { ...this.state.context };
    
    if (key) {
      // 清除特定键
      delete contextUpdate[key];
    } else {
      // 清除所有上下文
      Object.keys(contextUpdate).forEach(k => {
        delete contextUpdate[k];
      });
    }
    
    this.update({
      context: contextUpdate
    });
  }
  
  /**
   * 添加待执行操作
   * @param {Object} operation 操作对象
   */
  addPendingOperation(operation) {
    if (!operation) return;
    
    const pendingUpdate = [...this.state.pending, operation];
    
    this.update({
      pending: pendingUpdate
    });
    
    // 发出操作添加事件
    this.emit('operation:added', {
      operation,
      pendingCount: pendingUpdate.length,
      timestamp: Date.now()
    });
  }
  
  /**
   * 移除待执行操作
   * @param {number} index 操作索引
   */
  removePendingOperation(index) {
    if (index < 0 || index >= this.state.pending.length) return;
    
    const pendingUpdate = [...this.state.pending];
    const removed = pendingUpdate.splice(index, 1)[0];
    
    this.update({
      pending: pendingUpdate
    });
    
    // 发出操作移除事件
    this.emit('operation:removed', {
      operation: removed,
      index,
      pendingCount: pendingUpdate.length,
      timestamp: Date.now()
    });
  }
  
  /**
   * 清除所有待执行操作
   */
  clearPendingOperations() {
    const count = this.state.pending.length;
    
    this.update({
      pending: []
    });
    
    // 发出操作清除事件
    this.emit('operations:cleared', {
      count,
      timestamp: Date.now()
    });
  }
  
  /**
   * 重置会话
   */
  reset() {
    // 保存之前的状态
    const previousState = { ...this.state };
    
    // 创建新的会话状态
    this.state = new SessionState();
    
    // 生成新的会话ID
    this.sessionId = this._generateSessionId();
    
    // 发出会话重置事件
    this.emit('session:reset', {
      previousState,
      timestamp: Date.now()
    });
  }
  
  /**
   * 更新最近笔记列表
   * @param {Object} noteInfo 笔记信息
   * @private
   */
  _updateRecentNotes(noteInfo) {
    // 移除已存在的相同笔记
    const existingIndex = this.state.recentNotes.findIndex(note => note.path === noteInfo.path);
    if (existingIndex !== -1) {
      this.state.recentNotes.splice(existingIndex, 1);
    }
    
    // 添加到列表开头
    this.state.recentNotes.unshift(noteInfo);
    
    // 限制列表大小
    if (this.state.recentNotes.length > 10) {
      this.state.recentNotes = this.state.recentNotes.slice(0, 10);
    }
  }
  
  /**
   * 添加历史记录
   * @param {string} type 记录类型
   * @param {*} data 记录数据
   * @private
   */
  _addToHistory(type, data) {
    // 创建历史记录
    const historyItem = {
      type,
      data,
      timestamp: Date.now()
    };
    
    // 添加到历史记录
    this.state.history.unshift(historyItem);
    
    // 限制历史记录大小
    this.state.history = this.state.history.slice(0, this.options.maxHistory);
    
    // 清理过期记录
    this._cleanupHistory();
  }
  
  /**
   * 清理过期历史记录
   * @private
   */
  _cleanupHistory() {
    const now = Date.now();
    const expiry = this.options.historyExpiry;
    
    this.state.history = this.state.history.filter(item => {
      return now - item.timestamp < expiry;
    });
  }
  
  /**
   * 生成会话ID
   * @returns {string} 会话ID
   * @private
   */
  _generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
  }
}

module.exports = {
  SessionManager,
  SessionState
};
