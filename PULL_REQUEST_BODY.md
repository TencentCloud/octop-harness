## What

在 octop-harness 侧实现 `request_structured_form` 工具：Agent 通过声明式 schema 向人类收集类型安全的表单输入，由网关层在校验层拦截非法数据。

支持四类字段——`integer`、`string`、`boolean`、`enum`——每类携带各自的声明式约束与可选默认值。

## Why

现有 `ask_user` 是自由文本。Agent 问「要几个副本？」，用户回「跟上次一样三个吧，不过先上两个」，最终取 3 还是 2 取决于模型当次的理解，不可复现、不可测试；约束也无法在协议层表达，越界值要到执行失败才暴露。

本 PR 把类型交给 schema、把合法性校验前移到提交时刻、把多轮澄清压缩成一次交互。

## How

| 文件 | 操作 | 职责 |
|---|---|---|
| `tools/structured_form.py` | 修改 | 四类字段 schema + 请求/响应信封 + 提交校验 + 工具函数 + headless fallback |
| `tools/__init__.py` | 修改 | 导出 `STRUCTURED_FORM_TOOL_NAME`、`request_structured_form` |
| `agent.py` | 修改 | 按开关注册工具、`_structured_form_active()`、HITL 中断（`respond` / `cancel`） |
| `config/__init__.py` | 修改 | `structured_form_enabled: bool = True` |
| `tests/unit/test_structured_form.py` | 新增 | 65 项单测 |

### 核心设计

**字段类型系统**。`StructuredFieldBase` 承载 key/label/required/nullable/description，派生四类字段：

- `integer`：min/max 含边界、step=1
- `string`：min_length/max_length（≤4096）、pattern（编译期校验）、multiline
- `boolean`：nullable 强制 false，null 不表达布尔语义
- `enum`：options 非空、去重、单项 ≤120 字符、≤64 项

**默认值受约束校验**。每个字段都会用自身约束校验 `default`，非法默认值在构造期就报错，避免渲染端预填一个永远提交不过的值。

**逐字段错误定位**。`validate_structured_form_submission(fields, values)` 返回 `list[StructuredFormFieldError]`，每条错误携带 `field` 与 `message`。这样渲染端能就地高亮出错控件，而不是整单打回让用户重填 12 个字段去修其中 1 个。

**构造期拦截**。required 与 nullable 互斥、max < min、length 倒置、pattern 编译失败、default 越界或不在 options 内、key 不匹配 `^[a-z][a-z0-9_]*$`，全部在 schema 构造时拒绝。

**工具注册**。按 `structured_form_enabled` 配置挂载，与 `ask_user` 独立。Headless 模式下立即返回提示文本，不阻塞、不编造字段值。

## 顺带修复的两处既有缺陷

1. **`timeout_seconds` 从未绑定到工具函数**。它早在 `StructuredFormRequestArgs` 中声明，但工具签名缺少该参数，任何带超时的真实调用都会抛 `TypeError`。此前测试只验证 schema 与校验纯函数，从未真正 invoke 工具，因此漏检。已补参数并新增回归测试锁定。
2. **`tests/test_agent.py` 的 `interrupt_on` 断言过期**。上一提交已在 agent 中注册 `request_structured_form` 中断，但断言仍只期望 `ask_user_question`，导致该用例在基线上即为红。已更新为两者共存，并补充注释说明它们同属协作通道而非审批门禁。

## 当前范围与后续计划

已交付：四类字段、字段级约束、默认值、逐字段提交校验、headless 兜底、HITL 中断注册。

后续（不在本 PR 范围）：

- P1：`number`（浮点）、字段分组/分页、只读回显字段
- P2：表单模板复用、条件显示、多语言 label、敏感字段脱敏

常量限制：最多 64 字段、最多 16 context key、timeout 30–3600 秒、提交文本 4096 字符上限。

## Test

### 单元测试

`tests/unit/test_structured_form.py`：**65 passed**。覆盖协议常量、四类字段构造期约束（含 default 越界各组合）、请求信封边界（字段数上限、context 键上限、timeout 边界通过/拒绝、重复 key、未知类型、错误协议、非法 form_id、多余字段）、响应信封、提交校验全场景、工具表面（含四类字段混合 invoke 与带 timeout 的 invoke）。

### 回归

- 全量 `pytest tests`（排除三个已核实的基线环境失败文件）：**2084 passed, 22 skipped, 0 failed**
- `TestBootstrapMiddlewareWiring` + 新单测：**71 passed**
- ruff：源码与测试全部 **All checks passed**

已核实的三个基线失败文件（暂存本次改动后同样失败，与本改动无关）：`test_memory_call_policy.py`、`test_observability/test_langfuse.py`、`test_session_header.py`。

### 待办

- 端到端测试仍缺（真实模型 → 触发中断 → 提交表单 → 恢复执行）
- **协议对齐待完成**：配套网关 PR 的校验器目前仍限定 integer 字段，需同步扩展后才能支持本 PR 新增的三类字段

## Checklist

- [x] 测试用例已添加并通过 — 65/65
- [x] 与 ask_user 独立，不破坏现有行为
- [x] integer 字段的既有语义保持不变，旧表单请求无需修改
- [ ] CI — 待上游触发
- [ ] 网关侧协议同步 — 待配套 PR 扩展字段类型

## Related

- 网关侧 PR：labspc/Octop → `feat/hitl-postgres-store`（含信封校验、表单生命周期、PG 持久化）
- 迁移工具 PR：labspc/Octop → `feat/db-migrate-to-pg`
- Base: octop-harness v1.0.9
