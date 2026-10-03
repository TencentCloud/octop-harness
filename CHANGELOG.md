# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
