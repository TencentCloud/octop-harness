"""Read-only access to a local resource directory."""

from __future__ import annotations

from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import DeleteResult, EditResult, FileInfo, FileUploadResponse, WriteResult


class ReadOnlyResourceBackend(FilesystemBackend):
    """Expose a local directory through virtual paths without allowing mutations."""

    def __init__(self, root_dir: str | Path) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=True)

    def ls_info(self, path: str) -> list[FileInfo]:
        """Return directory entries in the legacy listing format."""
        result = self.ls(path)
        if result.error:
            raise OSError(result.error)
        return result.entries or []

    async def als_info(self, path: str) -> list[FileInfo]:
        """Asynchronously return directory entries."""
        result = await self.als(path)
        if result.error:
            raise OSError(result.error)
        return result.entries or []

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error="permission_denied: read-only backend")

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return EditResult(error="permission_denied: read-only backend")

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error="permission_denied: read-only backend")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]

    def mkdir_path(self, path: str) -> None:
        raise PermissionError("read-only backend")

    def delete_path(self, path: str) -> None:
        raise PermissionError("read-only backend")

    def move_path(self, src: str, dest: str) -> None:
        raise PermissionError("read-only backend")

    async def amkdir_path(self, path: str) -> None:
        self.mkdir_path(path)

    async def adelete_path(self, path: str) -> None:
        self.delete_path(path)

    async def amove_path(self, src: str, dest: str) -> None:
        self.move_path(src, dest)
