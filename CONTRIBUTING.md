# 贡献指南

感谢你对本项目的兴趣！我们欢迎各种形式的贡献，包括但不限于代码改进、Bug 修复、文档完善和新功能开发。

## 如何开始

### 1. Fork 和 Clone

```bash
git clone https://github.com/YOUR_USERNAME/A-Stock-Advisor.git
cd A-Stock-Advisor
```

### 2. 环境搭建

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
cd Finance
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -e ".[dev]"

# 配置环境变量
cd Financial-MCP-Agent
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 3. 运行测试

```bash
cd Finance/Financial-MCP-Agent
python -m pytest tests/ -v
```

## 贡献流程

### 提交 Issue

- **Bug 报告**：使用 Bug Report 模板，提供复现步骤和环境信息
- **功能建议**：使用 Feature Request 模板，描述功能需求和使用场景
- **问题咨询**：使用 Q&A 模板

### 提交 Pull Request

1. **创建分支**：从 `main` 分支创建你的功能分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

2. **编写代码**：遵循以下规范

3. **提交更改**：使用清晰的 commit message

```bash
git commit -m "feat: add new feature description"
# 或
git commit -m "fix: resolve issue with ..."
```

4. **推送并创建 PR**：

```bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request。

## 代码规范

### Python 代码风格

- 使用 **Black** 格式化代码：`black src/ tests/`
- 使用 **Ruff** 检查代码：`ruff check src/ tests/`
- 行长度限制：100 字符
- 使用类型注解（Type Hints）
- 函数和类必须有 docstring

### 新增 Agent 规范

如果你要新增分析 Agent，必须遵守以下规则：

1. **必须输出 SignalPack**：每个分析 Agent 必须输出 `<SIGNAL_PACK>` 结构化 JSON
2. **使用 model_config**：不要硬编码模型名称，使用 `get_model_config_for_agent()`
3. **缓存隔离**：评估系统的修改必须使用 `cache_namespace="eval"`，不得污染生产缓存
4. **反幻觉设计**：区分「数据事实区」和「分析判断区」，使用分级语言处理缺失数据

### Commit Message 格式

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Type** 可选值：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

**Scope** 可选值：
- `agent`: Agent 相关
- `eval`: 评估系统
- `ui`: 前端界面
- `api`: API 相关
- `cache`: 缓存相关
- `pool`: 股票池相关

## 审查流程

1. 所有 PR 需要至少 1 个维护者 review
2. CI 测试必须通过
3. 代码覆盖率不应显著下降
4. 请保持 PR 范围清晰，避免在一个 PR 中包含不相关的更改

## 许可证

提交代码即表示你同意将你的贡献按照项目的 LICENSE 授权给本项目。

## 联系方式

如有问题，请通过 GitHub Issue 或 Discussion 与我们联系。
