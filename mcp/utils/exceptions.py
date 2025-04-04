"""
异常模块 - 定义项目中使用的自定义异常
"""


class MCPError(Exception):
    """MCP平台的基础异常类"""
    pass


class APIError(MCPError):
    """API调用相关异常"""
    
    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class ModelError(MCPError):
    """模型相关异常"""
    pass


class ConfigurationError(MCPError):
    """配置相关异常"""
    pass


class AuthenticationError(APIError):
    """认证相关异常"""
    pass


class RateLimitError(APIError):
    """速率限制异常"""
    pass


class InvalidRequestError(APIError):
    """无效请求异常"""
    pass


class APIConnectionError(APIError):
    """API连接异常"""
    pass


class APITimeoutError(APIError):
    """API超时异常"""
    pass


class CacheError(MCPError):
    """缓存相关异常"""
    pass


class ObsidianError(MCPError):
    """Obsidian相关异常"""
    pass
