# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Streamed content blocks (Anthropic thinking / text, OpenAI Responses) no
  longer crash chunk projection: list-typed ``AIMessageChunk.content`` is
  normalized per block (thinking → reasoning, text → token) instead of raising
  ``TypeError`` in ``ThinkSplitter.feed``. Plain ``str`` content is unchanged.
- S3 backend resolution no longer aborts on unknown spec keys coming from
  free-form storage configs (Octop ``config_json``, e.g. ``s3_force_path_style``):
  unsupported keys are dropped with a warning whichever ``S3Config`` variant is
  constructed, and the community path-style booleans (``s3_force_path_style`` /
  ``force_path_style`` / ``path_style``) translate to ``addressing_style="path"``
  on the bundled boto3 backend. A requested ``addressing_style`` the third-party
  config cannot honour logs a targeted warning.

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
