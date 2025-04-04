# MCP平台缓存系统

缓存系统是MCP平台的核心组件之一，它能够有效减少API调用，降低成本，并提高响应速度。本文档介绍了缓存系统的架构、使用方法和管理方式。

## 1. 缓存架构

MCP平台采用多层缓存架构，包括：

### 1.1 内存缓存

- 存储在应用程序内存中，速度最快
- 应用重启后数据会丢失
- 适合临时和高频访问数据

### 1.2 磁盘缓存

- 存储在本地文件系统中，持久化保存
- 应用重启后数据仍然保留
- 适合长期存储和大量数据

### 1.3 缓存管理器

缓存管理器(`CacheManager`)协调不同层级的缓存，提供统一的接口进行访问和管理：

- 自动在内存缓存中查找数据
- 如果内存缓存未命中，则查找磁盘缓存
- 从磁盘缓存获取的数据会自动添加到内存缓存

## 2. 配置缓存

缓存系统通过`.env`文件进行配置：

```
# 缓存配置
CACHE_ENABLED=true            # 是否启用缓存
CACHE_DIR=./cache             # 缓存目录路径
CACHE_EXPIRY=86400            # 缓存过期时间（秒），默认为1天
```

## 3. 使用缓存

### 3.1 通过API客户端使用

缓存已内置到所有API客户端中，无需额外操作即可自动使用：

```python
from mcp.api.client import MCPClient

# 创建客户端
client = MCPClient()

# 所有生成调用都会自动使用缓存
response = client.generate("你好，请介绍一下自己")

# 相同的请求会从缓存返回结果，而不是重新调用API
same_response = client.generate("你好，请介绍一下自己")
```

### 3.2 手动使用缓存

也可以直接使用缓存管理器：

```python
from mcp.utils.cache import cache_manager

# 获取缓存值
value = cache_manager.get("my_key")

# 设置缓存值
cache_manager.set("my_key", {"data": "some value"}, ttl=3600)  # 1小时过期

# 检查键是否存在
exists = cache_manager.exists("my_key")

# 删除键
cache_manager.delete("my_key")
```

### 3.3 使用缓存装饰器

对于需要缓存结果的函数，可以使用缓存装饰器：

```python
from mcp.utils.cache import cache_manager

# 缓存函数结果，默认使用内存缓存
@cache_manager.cache_function(ttl=3600)
def expensive_calculation(param1, param2):
    # 耗时计算...
    return result

# 使用磁盘缓存
@cache_manager.cache_function(ttl=86400, cache_name="disk")
def very_expensive_calculation(param1, param2):
    # 更耗时的计算...
    return result
```

## 4. 管理缓存

### 4.1 通过命令行工具

MCP平台提供了命令行工具管理缓存：

```bash
# 显示缓存统计信息
mcp cache --info

# 显示特定缓存的统计信息
mcp cache --info --name memory

# 查看缓存键列表
mcp cache --keys

# 清空所有缓存
mcp cache --clear

# 清空特定缓存
mcp cache --clear --name disk

# 删除特定缓存键
mcp cache --delete-key "deepseek:a1b2c3d4e5f6"
```

### 4.2 通过代码

也可以通过代码管理缓存：

```python
from mcp.utils.cache import cache_manager

# 获取缓存统计
stats = cache_manager.get_stats()
print(f"内存缓存命中率: {stats['caches']['memory']['hit_ratio']:.2%}")

# 清空所有缓存
cache_manager.clear()

# 获取特定缓存实例
memory_cache = cache_manager.get_cache("memory")
keys = memory_cache.get_keys()
```

## 5. 性能考虑

缓存系统默认配置适合大多数场景，但在特定情况下可能需要调整：

- **内存缓存大小**：默认保存1000项，可根据可用内存调整
- **磁盘缓存大小**：默认保存10000项，可根据磁盘空间调整
- **过期时间**：根据数据更新频率调整，对于静态内容可以设置更长时间

## 6. 最佳实践

- 对于固定不变的请求，尽量使用相同的参数以提高缓存命中率
- 对于不希望缓存的敏感信息，可以直接使用API客户端的低级API调用
- 定期清理磁盘缓存释放空间，尤其是在长期运行的应用中
- 使用命令行工具监控缓存命中率，以评估缓存效果
