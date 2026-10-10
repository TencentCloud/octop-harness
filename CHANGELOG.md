# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 新增

- 插件系统新增 `channel` 插件类型：`kind: channel` 插件可在 `setup()` 中通过 `ctx.channel(kind, channel_cls, label=..., fields=...)` 贡献渠道实现；`PluginRegistry.all_channels()` 汇总全部插件渠道注册（同 kind 冲突时先注册者优先）。

- 记忆辅助调用可配置客户端兜底 `memory_aux_max_tokens`（提炼预算仍走 `memory_extract_max_tokens`），以及 OpenAI 兼容端点的 `memory_aux_extra_body`。显式 `extra_body` 覆盖 `memory_aux_thinking` 生成的同名键。

## [1.0.2] - 2026-10-09

### 修复

- 将 `deepagents` 下限提高到 `>=0.7.16`，接入上游同路径并行 `edit_file` / `write_file` 防护（langchain-ai/deepagents#6446）。
- 记忆辅助调用可单独配置读取超时。提炼输出预算写入 memory runtime 的 `extraction.max_tokens`（提取器总会自带预算，客户端兜底覆盖不到）。OpenAI 兼容端点在 `auto` 下对 Qwen3 / QwQ / DeepSeek-R / reasoner 默认关闭深度思考（TencentCloud/Octop#1360）。
- 小时增量 vacuum 预算提高到 5000 页，大批删除后的 freelist 可在数小时内回收，而不是数周。

## [1.0.1] - 2026-10-03

### 新增

- PII 可识别大陆手机号与 18 位居民身份证。
- 子树专家支持 `explicit_virtual_paths`，由调用方自行转换虚拟路径。
- 新增讯飞星辰 Token Plan 供应商预设。

### 修复

- 子 Agent 仅在真正重名时告警。
- 模型和路径错误回传给 Agent，不再整轮中止。
- 修复 S3 / Postgres workspace 与 deepagents 0.7 的兼容，文件不再摊平到存储根。
- Docker 每次执行刷新全局环境。
- 流式 thinking / text 块不再导致协议投影崩溃。

### 变更

- 发版时同步检查多语言 README 版本。
### Added

- PII protection detects mainland China mobile numbers (common 13-19
  prefixes, optional ``+86`` / ``0086``) and 18-character resident IDs with
  a known province code, 1900-2099 birth date, and a valid checksum.
  Existing ``pii_strategy`` / ``pii_surfaces`` apply; 15-digit IDs and other
  countries’ numbers are out of scope.
- Optional ``explicit_virtual_paths`` for subtree experts: shell command,
  environment, and stdout are left unparsed; callers convert one virtual
  absolute path through ``virtual_to_native_path``. Default stays off.
- Bundled provider templates: iFlytek Astron MaaS Token Plan preset
  `iflytek-astron-token-plan` (Spark-X2.5 first; Kimi-K2.6 and Qwen3.5/3.6 marked
  image-capable), with an `iflytek.svg` logo.

### Fixed

- Workspace subagent loading no longer logs a spurious ``Duplicate subagent
  name`` warning when the same ``.octop/agents/*.md`` definition is discovered
  through both the legacy ``agents/`` root and the canonical
  ``system_files_path`` root; the warning now only fires for genuine name
  collisions between different files.
- Leftover Windows absolute paths in filesystem tool calls (``D:\\octop-data\\…``)
  are rewritten onto the current storage root when the same suffix exists, and
  still soft-fail to the model when they do not — they no longer abort the turn
  as a generic model-call failure.
- S3 backend resolution always uses the bundled boto3 implementation, which
  speaks deepagents 0.7 (`WriteResult` / `ReadResult` / `ls` / `glob` /
  `grep`). The `deepagents-backends` 0.2 S3 client is no longer preferred —
  it still passes the removed `files_update` keyword and cannot list or read
  under 0.7. Community path-style aliases (`s3_force_path_style` /
  `force_path_style` / `path_style`) map to `addressing_style="path"`.
  Custom endpoints default to SigV4 + path-style, disable optional CRC
  checksums, and fall back ListObjectsV2 → ListObjects / GetObject when the
  store rejects a header or signature flavour.
- Postgres backend specs accept a libpq `connection_string` / `dsn` and drop
  unknown keys instead of raising `TypeError` on `PostgresConfig`.
- Postgres workspace I/O uses a bundled `psycopg` backend that speaks
  deepagents 0.7 (`WriteResult` / `ReadResult` / `ls` / `glob` / `grep`).
  Probe performs a real write→read→delete round-trip; the table is created
  on first use.
- COS / S3 / OSS / OBS / Postgres workspaces no longer flatten files onto
  the storage root. Agent files live under `/.octop/workspaces/<id>/` so
  listing `/` shows workspace folders, then the workspace tree. Host
  paths such as `~/.octop/agents/<id>` are remapped onto that virtual
  folder and are not stored as object keys.
- Docker execution reloads the global environment file even when a same-size edit or atomic replacement preserves its modification timestamp, avoiding stale injected values.
- Streamed content blocks (Anthropic thinking / text, OpenAI Responses) no
  longer crash chunk projection: list-typed ``AIMessageChunk.content`` is
  normalized per block (thinking → reasoning, text → token) instead of raising
  ``TypeError`` in ``ThinkSplitter.feed``. Plain ``str`` content is unchanged.

## [1.0.0] - 2026-09-24

### 新增

- 首个 1.0.0 正式版本发布到 PyPI。

### 变更

- 对齐 Octop：引入 `develop` 集成分支策略；禁止直推 `main`/`develop`；发版后由 `sync-main-to-develop.yml` 同步；新增 `/publish` skill（发版同步 CHANGELOG / README）。



### Fixed

- Built-in tools soft-fail instead of aborting the agent turn: ``web_fetch``
  (non-http(s) / oversized body), ``current_time`` (unknown timezone),
  ``desktop_screenshot`` (capture ``RuntimeError``), ``acp_runner`` (missing
  ``thread_id`` on close/stream), and local ``execute`` (non-positive timeout).
- Filesystem tools: ``Path ... outside root directory`` ``ValueError`` from
  deepagents ``FilesystemBackend`` is softened by ``FilesystemGuardMiddleware``
  (always mounted) into a ``ToolMessage(status="error")`` so ToolNode does not
  abort the agent turn; deny-path rules still apply when permissions are set.
