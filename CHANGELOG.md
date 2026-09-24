# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Built-in tools soft-fail instead of aborting the agent turn: ``web_fetch``
  (non-http(s) / oversized body), ``current_time`` (unknown timezone),
  ``desktop_screenshot`` (capture ``RuntimeError``), ``acp_runner`` (missing
  ``thread_id`` on close/stream), and local ``execute`` (non-positive timeout).
- Filesystem tools: ``Path ... outside root directory`` ``ValueError`` from
  deepagents ``FilesystemBackend`` is softened by ``FilesystemGuardMiddleware``
  (always mounted) into a ``ToolMessage(status="error")`` so ToolNode does not
  abort the agent turn; deny-path rules still apply when permissions are set.
