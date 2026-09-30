"""Tests for ``octop_harness.backends.resolve_backend``."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from deepagents.backends import CompositeBackend, LocalShellBackend
from deepagents.middleware.filesystem import supports_execution

from octop_harness.backends import (
    DEFAULT_BACKEND_SPEC,
    MountedCompositeBackend,
    _postgres_config_kwargs,
    resolve_backend,
    spec_supports_execution,
)
from octop_harness.backends.workspace import BackendWorkspace


class TestDefaultSpec:
    def test_default_backend_spec(self) -> None:
        assert DEFAULT_BACKEND_SPEC == {
            "type": "local_shell",
            "root_dir": "/",
            "virtual_mode": True,
        }

    def test_none_resolves_to_local_shell_root(self) -> None:
        """resolve_backend(None) without workspace keeps a bare host-rooted shell."""
        backend = resolve_backend(None)
        assert isinstance(backend, LocalShellBackend)

    def test_none_with_workspace_scopes_artifacts(self, tmp_path: Path) -> None:
        backend = resolve_backend(None, workspace_dir=tmp_path)
        assert isinstance(backend, MountedCompositeBackend)
        assert backend.artifacts_root == str(tmp_path.resolve())
        assert isinstance(backend.default, LocalShellBackend)
        assert str(backend.cwd) == "/"
        assert supports_execution(backend)

    def test_host_root_spec_with_workspace_scopes_artifacts(self, tmp_path: Path) -> None:
        backend = resolve_backend(dict(DEFAULT_BACKEND_SPEC), workspace_dir=tmp_path)
        assert isinstance(backend, MountedCompositeBackend)
        assert backend.artifacts_root == str(tmp_path.resolve())

        history = tmp_path / "conversation_history" / "thread.md"
        result = backend.write(str(history), "context summary")
        assert result.error is None
        assert history.read_text(encoding="utf-8") == "context summary"

    def test_system_files_path_scopes_artifacts_under_prefix(self, tmp_path: Path) -> None:
        backend = resolve_backend(
            dict(DEFAULT_BACKEND_SPEC),
            workspace_dir=tmp_path,
            system_files_path=".octop",
        )
        assert isinstance(backend, MountedCompositeBackend)
        assert backend.artifacts_root == str((tmp_path / ".octop").resolve())

        history = tmp_path / ".octop" / "conversation_history" / "thread.md"
        result = backend.write(str(history), "context summary")
        assert result.error is None
        assert history.read_text(encoding="utf-8") == "context summary"

    def test_custom_root_dir_scopes_artifacts_via_virtual_path(self, tmp_path: Path) -> None:
        """Non-host root_dir must still offload under the workspace (virtual path)."""
        root = tmp_path / "home"
        ws = root / ".octop" / "workspaces" / "NBR8CP"
        ws.mkdir(parents=True)

        backend = resolve_backend(
            {
                "type": "local_shell",
                "root_dir": str(root),
                "virtual_mode": True,
            },
            workspace_dir=ws,
            system_files_path=".octop",
        )
        assert isinstance(backend, MountedCompositeBackend)
        assert backend.artifacts_root == "/.octop/workspaces/NBR8CP/.octop"

        history_key = f"{backend.artifacts_root}/conversation_history/thread.md"
        result = backend.write(history_key, "context summary")
        assert result.error is None
        host_file = ws / ".octop" / "conversation_history" / "thread.md"
        assert host_file.read_text(encoding="utf-8") == "context summary"

    def test_host_root_composite_keeps_workspace_mutations(self, tmp_path: Path) -> None:
        backend = resolve_backend(dict(DEFAULT_BACKEND_SPEC), workspace_dir=tmp_path)
        workspace = BackendWorkspace(backend, tmp_path)
        workspace.mkdir("source")
        source = tmp_path / "source" / "note.txt"
        source.write_text("ok", encoding="utf-8")
        workspace.move("source", "moved")
        assert (tmp_path / "moved" / "note.txt").read_text(encoding="utf-8") == "ok"
        workspace.delete("moved")
        assert not (tmp_path / "moved").exists()


class TestSpecSupportsExecution:
    """``spec_supports_execution`` must predict ``supports_execution`` from the spec alone."""

    def test_none_is_shell_capable(self) -> None:
        assert spec_supports_execution(None) is True

    def test_local_shell_string(self) -> None:
        assert spec_supports_execution("local_shell") is True

    def test_filesystem_string(self) -> None:
        assert spec_supports_execution("filesystem") is False

    def test_composite_follows_default_not_routes(self) -> None:
        spec = {
            "type": "composite",
            "default": "filesystem",
            "routes": {"/tmp/": "local_shell"},
        }
        assert spec_supports_execution(spec) is False

    def test_composite_with_shell_default(self) -> None:
        spec = {"type": "composite", "default": {"type": "local_shell"}}
        assert spec_supports_execution(spec) is True

    def test_backend_instance(self, tmp_path: Path) -> None:
        backend = resolve_backend({"type": "local_shell", "root_dir": str(tmp_path)})
        assert spec_supports_execution(backend) is True

    @pytest.mark.parametrize(
        "spec",
        [
            None,
            "local_shell",
            "filesystem",
            "state",
            {"type": "local_shell", "root_dir": "/"},
            {"type": "composite", "default": "filesystem", "routes": {"/x/": "local_shell"}},
        ],
    )
    def test_matches_resolved_backend(self, spec: object, tmp_path: Path) -> None:
        resolved = resolve_backend(spec, workspace_dir=tmp_path)
        assert spec_supports_execution(spec) is supports_execution(resolved)


class TestStringForms:
    def test_filesystem_string(self, tmp_path: Path) -> None:
        from deepagents.backends import FilesystemBackend

        backend = resolve_backend("filesystem", workspace_dir=tmp_path)
        assert isinstance(backend, FilesystemBackend)

    def test_state_string(self) -> None:
        from deepagents.backends import StateBackend

        backend = resolve_backend("state")
        assert isinstance(backend, StateBackend)


class TestDictForms:
    def test_filesystem_with_explicit_root(self, tmp_path: Path) -> None:
        from deepagents.backends import FilesystemBackend

        backend = resolve_backend({"type": "filesystem", "root_dir": str(tmp_path)})
        assert isinstance(backend, FilesystemBackend)

    def test_local_shell_with_explicit_workspace_no_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Passing ``workspace_dir`` (non-'/') should NOT trigger the warning."""
        from deepagents.backends import LocalShellBackend

        with caplog.at_level(logging.WARNING):
            backend = resolve_backend("local_shell", workspace_dir=tmp_path)
        assert isinstance(backend, LocalShellBackend)
        assert not any("rooted at '/'" in rec.message for rec in caplog.records)

    def test_local_shell_default_root_warns(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Default root is '/', which DOES trigger the warning."""
        from deepagents.backends import LocalShellBackend

        with caplog.at_level(logging.WARNING):
            backend = resolve_backend("local_shell")
        assert isinstance(backend, LocalShellBackend)
        assert any("rooted at '/'" in rec.message for rec in caplog.records)

    def test_local_shell_root_slash_warns(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Explicitly setting root_dir='/' triggers the warning (includes workspace)."""
        with caplog.at_level(logging.WARNING):
            resolve_backend(
                {"type": "local_shell", "root_dir": "/"},
                workspace_dir=tmp_path,
            )
        matching = [rec.message for rec in caplog.records if "rooted at '/'" in rec.message]
        assert matching
        assert f"workspace={tmp_path}" in matching[0]

    def test_local_shell_explicit_root_no_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from deepagents.backends import LocalShellBackend

        from octop_harness.backends.bwrap_shell import (
            BubbledLocalShellBackend,
            HarnessLocalShellBackend,
            can_use_bubbled_shell,
        )

        with caplog.at_level(logging.WARNING):
            backend = resolve_backend({"type": "local_shell", "root_dir": str(tmp_path)})
        assert isinstance(backend, HarnessLocalShellBackend)
        assert isinstance(backend, LocalShellBackend)
        assert isinstance(backend, BubbledLocalShellBackend) is can_use_bubbled_shell(
            virtual_mode=True,
            root_dir=tmp_path,
        )
        if isinstance(backend, BubbledLocalShellBackend):
            assert backend._bwrap_path
        # Specifically the "rooted at /" warning should not fire.
        assert not any("rooted at '/'" in rec.message for rec in caplog.records)


class TestComposite:
    def test_composite_recursively_resolves_subspecs(self, tmp_path: Path) -> None:
        from deepagents.backends import FilesystemBackend, StateBackend

        backend = resolve_backend(
            {
                "type": "composite",
                "default": "state",
                "routes": {
                    "/workspace/": {"type": "filesystem", "root_dir": str(tmp_path)},
                },
            },
        )
        assert isinstance(backend, MountedCompositeBackend)
        assert isinstance(backend, CompositeBackend)
        # Composite stores the resolved subbackends — peek inside to make sure
        # the recursion did the right thing.
        assert isinstance(backend.default, StateBackend)  # type: ignore[attr-defined]
        routes = backend.routes  # type: ignore[attr-defined]
        assert isinstance(routes["/workspace/"], FilesystemBackend)

    def test_composite_with_workspace_defaults_artifacts_root(self, tmp_path: Path) -> None:
        from deepagents.backends import LocalShellBackend

        backend = resolve_backend(
            {
                "type": "composite",
                "default": {"type": "local_shell", "root_dir": "/", "virtual_mode": True},
                "routes": {},
            },
            workspace_dir=tmp_path,
        )
        assert isinstance(backend, MountedCompositeBackend)
        assert backend.artifacts_root == str(tmp_path.resolve())
        # Sub-backend must not be double-wrapped.
        assert isinstance(backend.default, LocalShellBackend)
        assert str(backend.cwd) == "/"

    def test_composite_explicit_artifacts_root_preserved(self, tmp_path: Path) -> None:
        custom = str((tmp_path / "custom-artifacts").resolve())
        backend = resolve_backend(
            {
                "type": "composite",
                "default": "state",
                "routes": {},
                "artifacts_root": custom,
            },
            workspace_dir=tmp_path,
        )
        assert backend.artifacts_root == custom

    def test_composite_missing_default_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a 'default'"):
            resolve_backend({"type": "composite", "routes": {}})

    def test_composite_routes_must_be_dict(self) -> None:
        with pytest.raises(ValueError, match="routes"):
            resolve_backend({"type": "composite", "default": "state", "routes": []})


class TestErrors:
    def test_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend type"):
            resolve_backend("teleporter")

    def test_dict_without_type(self) -> None:
        with pytest.raises(ValueError, match="'type' key"):
            resolve_backend({"root_dir": "/"})

    def test_invalid_spec_kind(self) -> None:
        with pytest.raises(ValueError, match=r"(Backend spec must be|BackendProtocol)"):
            resolve_backend(42)

    def test_instance_passed_through(self) -> None:
        from deepagents.backends import StateBackend

        sentinel = StateBackend()
        assert resolve_backend(sentinel) is sentinel


class TestRemoteBackendsImportError:
    """When ``deepagents-backends`` isn't importable, the factory must surface
    a helpful error rather than a generic ``ModuleNotFoundError``."""

    def test_s3_falls_back_to_s3_compat_without_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # S3 now falls back to S3CompatBackend (boto3) instead of raising ImportError.
        pytest.importorskip("boto3", reason="boto3 not installed; skipping S3CompatBackend fallback test")
        from octop_harness.backends.s3_backend import S3CompatBackend

        monkeypatch.setitem(sys.modules, "deepagents_backends", None)
        backend = resolve_backend(
            {
                "type": "s3",
                "bucket": "test-bucket",
                "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            }
        )
        assert isinstance(backend, S3CompatBackend)

    def test_postgres_without_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "deepagents_backends", None)
        with pytest.raises(ImportError, match="octop-harness\\[remote-backends\\]"):
            resolve_backend({"type": "postgres", "dsn": "postgresql://x"})


class TestPostgresConfigKwargs:
    """``_postgres_config_kwargs`` accepts a whole libpq URI as well as split fields.

    Exercised against a stub dataclass so the mapping is covered even where
    ``deepagents-backends`` (the optional extra) is not installed. URI parsing
    itself needs ``psycopg``, which ships with that extra.
    """

    @pytest.fixture(autouse=True)
    def _require_psycopg(self) -> None:
        pytest.importorskip(
            "psycopg",
            reason="psycopg not installed; skipping libpq connection-string mapping test",
        )

    @dataclass
    class _StubConfig:
        host: str = "localhost"
        port: int = 5432
        database: str = "deepagents"
        user: str = "postgres"
        password: str = ""
        table: str = "files"
        schema: str = "public"
        min_pool_size: int = 5
        max_pool_size: int = 20
        max_idle_seconds: float = 300.0
        connection_timeout: float = 30.0
        sslmode: str = "prefer"

    def test_connection_string_uri_is_parsed(self) -> None:
        """dbname → database, port as int, and query params the config models."""
        resolved = _postgres_config_kwargs(
            {"connection_string": "postgresql://alice:s3cr3t@db.example.com:5433/mydb?sslmode=require"},
            self._StubConfig,
        )
        assert resolved["host"] == "db.example.com"
        assert resolved["port"] == 5433
        assert resolved["database"] == "mydb"
        assert resolved["user"] == "alice"
        assert resolved["password"] == "s3cr3t"
        assert resolved["sslmode"] == "require"

    def test_dsn_alias_is_accepted(self) -> None:
        resolved = _postgres_config_kwargs(
            {"dsn": "postgresql://bob:pw@localhost/reports"},
            self._StubConfig,
        )
        assert resolved["user"] == "bob"
        assert resolved["database"] == "reports"

    def test_non_uri_fields_survive_alongside_connection_string(self) -> None:
        """Knobs the URI cannot express (table / schema / pool sizing) are kept.

        Octop's adapter sets ``schema`` when the row's region is not ``public``.
        """
        resolved = _postgres_config_kwargs(
            {
                "connection_string": "postgresql://u:p@h/d",
                "schema": "harness",
                "max_pool_size": 42,
            },
            self._StubConfig,
        )
        assert resolved["schema"] == "harness"
        assert resolved["max_pool_size"] == 42
        assert resolved["host"] == "h"

    def test_empty_connection_string_does_not_leak_the_key(self) -> None:
        """A falsy URI must still be filtered, not passed through as a kwarg.

        ``""`` took the early return and handed ``kwargs`` over untouched, so
        ``PostgresConfig(connection_string="")`` raised the same TypeError this
        helper exists to prevent — reachable whenever a caller stores an empty
        string instead of omitting the field.
        """
        resolved = _postgres_config_kwargs(
            {"connection_string": "", "host": "h", "user": "u"},
            self._StubConfig,
        )
        assert "connection_string" not in resolved
        self._StubConfig(**resolved)

    def test_keys_without_a_slot_are_dropped_without_a_uri(self) -> None:
        """Unslotted keys are dropped on the split-field path too.

        The docstring promises this unconditionally, but only the URI branch
        filtered; a spec carrying e.g. ``previewable`` raised ``TypeError``
        inside ``PostgresConfig``.
        """
        resolved = _postgres_config_kwargs(
            {"host": "h", "previewable": True},
            self._StubConfig,
        )
        assert resolved == {"host": "h"}

    def test_invalid_uri_error_does_not_leak_password(self) -> None:
        """libpq echoes the offending token; the password must not reach the error."""
        with pytest.raises(ValueError) as excinfo:
            _postgres_config_kwargs(
                {"connection_string": "mysql://alice:s3cr3t@db.example.com/mydb"},
                self._StubConfig,
            )
        exc = excinfo.value
        assert "s3cr3t" not in str(exc)
        # With ``from exc`` the libpq error would sit on __cause__ and put the URI
        # back into any traceback dump, so assert on the chain, not just the message.
        assert exc.__cause__ is None
        assert exc.__suppress_context__ is True


class TestPostgresConnectionStringEndToEnd:
    """``resolve_backend`` wires the mapping into a real ``PostgresBackend``."""

    @pytest.fixture(autouse=True)
    def _require_extra(self) -> None:
        pytest.importorskip(
            "deepagents_backends",
            reason="deepagents-backends not installed; skipping postgres backend test",
        )

    def test_resolve_backend_accepts_connection_string(self) -> None:
        backend = resolve_backend(
            {
                "type": "postgres",
                "connection_string": "postgresql://alice:s3cr3t@db.example.com:5433/mydb",
                "schema": "harness",
            }
        )
        config = backend._config  # type: ignore[attr-defined]
        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.database == "mydb"
        assert config.schema == "harness"
