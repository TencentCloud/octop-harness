## What

在 octop-harness 侧实现 `request_structured_form` 工具——Agent 通过声明式 schema 向人类收集类型安全的整数输入，配合网关层在校验层拦截非法数据。

## Why

现有 `ask_user` 是自由文本。Agent 问「副本数？」，用户回「八个」，协议层束手无策。具体痛点见 octop PR。

## How

4 个文件，纯增量：

| 文件 | 操作 | 行数 | 职责 |
|---|---|---|---|
| `tools/structured_form.py` | 新增 | 110 | pydantic schema + request_structured_form 工具 + headless fallback |
| `tools/__init__.py` | 修改 | +6 | 导入/导出 STRUCTURED_FORM_TOOL_NAME + request_structured_form |
| `agent.py` | 修改 | +16 | 导入 → 按开关注册 → _structured_form_active() → HITL 中断 |
| `config/__init__.py` | 修改 | +7 | structured_form_enabled: bool = True |

### 核心设计

Schema 层：`StructuredIntegerField` 定义单个整数字段，pydantic model_validator 构造阶段拦截 required+nullable 互斥、max<min；`StructuredFormRequestArgs` 承载完整表单（form_id/title/fields 1-64 个，内置去重）；`StructuredFormResponse` 定义用户提交消息体。

工具注册：按 `structured_form_enabled` 配置挂载，与 `ask_user` 独立，Respond-only HITL 中断。Headless 模式下立即返回提示文本不阻塞。

## MVP 限制

- 字段类型仅 integer
- 常量限制：最多 64 字段、最多 16 context key、请求/响应各 32KB 上限

## Test

`test_structured_form.py` 25/25 全部通过（与 octop PR 共享同一测试脚本），覆盖 schema 边界、请求校验、响应校验、headless fallback、协议常量。

## Checklist

- [x] 🎯 最小修改原则 — 4 文件纯增量
- [x] 🧱 测试用例已添加并通过 — 25/25
- [x] 🎯 与 ask_user 独立，不破坏现有行为
- [ ] ✅ CI — 待上游触发

## Related

- 网关侧 PR: labspc/Octop → feature/structured-hitl-forms
- Base: octop-harness v1.0.9