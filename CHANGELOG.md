# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Bundled provider templates: iFlytek Astron MaaS Token Plan preset
  `iflytek-astron-token-plan` (Spark-X2.5 first; Kimi-K2.6 and Qwen3.5/3.6 marked
  image-capable), with an `iflytek.svg` logo.

### Fixed

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
