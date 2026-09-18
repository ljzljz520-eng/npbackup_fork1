#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# This file is part of npbackup

__intname__ = "npbackup.config_transaction"
__author__ = "Orsiris de Jong"
__copyright__ = "Copyright (C) 2022-2026 NetInvent"
__license__ = "GPL-3.0-only"
__build__ = "2026091801"


"""
Versioned, transactional configuration file storage.

Commit protocol (shared by configuration migrations and GUI saves):

1. The complete, already encrypted snapshot is validated by the caller.
2. Its bytes are written to a temporary file located in the SAME directory as
   the target, using restrictive permissions (0600).
3. The temporary file is fsync'ed. Its checksum is re-verified after the
   write, and a 0600 checksum sidecar file is written and fsync'ed.
4. The previous generation (exactly one encrypted copy and its checksum) is
   rotated, and the temporary files atomically replace the target via
   os.replace().
5. The directory is fsync'ed so the rename itself is persisted.

Load protocol:

The current generation and the retained previous generation are inspected.
A generation is considered valid when its file is readable, its checksum
sidecar matches and the caller provided parser accepts its bytes. The newest
valid generation is selected; when the current generation is unusable but the
previous one is valid, the previous generation is atomically re-published as
the current generation (without destroying the good copy) and an explicit
recovery result is returned.

Configuration files without a checksum sidecar (written by older npbackup
versions) are accepted as "legacy" generations for the current target and
become fully versioned on their next commit.
"""


import os
import sys
import hashlib
import secrets
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, List, Optional

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))


logger = getLogger()


# Exactly one previous generation is retained
GENERATION_SUFFIX = ".prev"
CHECKSUM_SUFFIX = ".sha256"
TEMP_FILE_PREFIX = ".{name}.tmp."
RESTRICTIVE_FILE_MODE = 0o600
# 1 MiB io chunks are plenty for configuration files
_IO_CHUNK_SIZE = 1024 * 1024

# Generation states
STATE_VALID = "valid"
STATE_LEGACY = "legacy"
STATE_CORRUPT_CHECKSUM = "corrupt_checksum"
STATE_INVALID_FORMAT = "invalid_format"
STATE_UNREADABLE = "unreadable"
STATE_ABSENT = "absent"

# Selection statuses
STATUS_CURRENT = "current"
STATUS_LEGACY = "legacy"
STATUS_RECOVERED_PREVIOUS = "recovered_previous"
STATUS_UNAVAILABLE = "unavailable"


class CommitResult:
    """
    Explicit outcome of a transactional commit / recovery publish.
    """

    def __init__(
        self,
        success: bool,
        config_file: Optional[Path] = None,
        checksum: Optional[str] = None,
        previous_checksum: Optional[str] = None,
        bytes_written: int = 0,
        recovered: bool = False,
        message: str = "",
    ) -> None:
        self.success = success
        self.config_file = Path(config_file) if config_file else None
        self.checksum = checksum
        self.previous_checksum = previous_checksum
        self.bytes_written = bytes_written
        self.recovered = recovered
        self.message = message

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        return (
            "<CommitResult success={} config_file={} checksum={} "
            "previous_checksum={} bytes_written={} recovered={}>".format(
                self.success,
                self.config_file,
                self.checksum,
                self.previous_checksum,
                self.bytes_written,
                self.recovered,
            )
        )


class GenerationInfo:
    """
    Inspection result of a single configuration generation.
    """

    def __init__(
        self,
        path: Path,
        state: str,
        checksum_expected: Optional[str] = None,
        checksum_actual: Optional[str] = None,
        size: int = 0,
        detail: str = "",
        data: Optional[bytes] = None,
    ) -> None:
        self.path = Path(path)
        self.state = state
        self.checksum_expected = checksum_expected
        self.checksum_actual = checksum_actual
        self.size = size
        self.detail = detail
        self.data = data

    def __repr__(self) -> str:
        return (
            "<GenerationInfo path={} state={} checksum_expected={} "
            "checksum_actual={} size={}>".format(
                self.path,
                self.state,
                self.checksum_expected,
                self.checksum_actual,
                self.size,
            )
        )


class SelectionResult:
    """
    Explicit outcome of a versioned configuration load.
    """

    def __init__(
        self,
        status: str,
        data: Optional[bytes] = None,
        parsed: Any = None,
        generations: Optional[List[GenerationInfo]] = None,
        recovery: Optional[CommitResult] = None,
        message: str = "",
    ) -> None:
        self.status = status
        self.data = data
        self.parsed = parsed
        self.generations = generations if generations is not None else []
        self.recovery = recovery
        self.message = message

    @property
    def recovered(self) -> bool:
        return self.status == STATUS_RECOVERED_PREVIOUS

    def __repr__(self) -> str:
        return "<SelectionResult status={} generations={}>".format(
            self.status, self.generations
        )


def checksum_sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _checksum_sidecar_path(path: Path) -> Path:
    return Path(str(path) + CHECKSUM_SUFFIX)


def _previous_generation_path(config_file: Path) -> Path:
    return Path(str(config_file) + GENERATION_SUFFIX)


def _fsync_fd(fd: int, label: str) -> None:
    """
    fsync a file descriptor. On some filesystems (eg network mounts), fsync
    may not be supported; the atomic rename still guarantees that readers
    never see a half written file, so we warn instead of failing.
    """
    try:
        os.fsync(fd)
    except OSError as exc:
        logger.warning(
            "Cannot fsync {} (durability may be reduced on this filesystem): {}".format(
                label, exc
            )
        )


def _fsync_directory(directory: Path) -> None:
    """
    fsync a directory so that renames/unlinks within it are persisted.
    Not supported on Windows.
    """
    if os.name == "nt":
        return
    dir_fd = None
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
        os.fsync(dir_fd)
    except OSError as exc:
        logger.warning(
            "Cannot fsync directory {} (durability may be reduced on this "
            "filesystem): {}".format(directory, exc)
        )
    finally:
        if dir_fd is not None:
            try:
                os.close(dir_fd)
            except OSError:
                pass


def _write_temp_file(directory: Path, target_name: str, data: bytes) -> Path:
    """
    Write data to a unique restrictive temporary file in directory, named after
    target_name so temp leftovers can be associated with a target.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    temp_path = None
    fd = None
    temp_name_prefix = TEMP_FILE_PREFIX.format(name=target_name)
    for _ in range(16):
        candidate = directory / "{}{}.{}".format(
            temp_name_prefix, os.getpid(), secrets.token_hex(8)
        )
        try:
            fd = os.open(str(candidate), flags, RESTRICTIVE_FILE_MODE)
            temp_path = candidate
            break
        except FileExistsError:
            continue
    if fd is None or temp_path is None:
        raise OSError("Cannot create unique temporary file in {}".format(directory))

    try:
        try:
            os.fchmod(fd, RESTRICTIVE_FILE_MODE)
        except OSError as exc:
            logger.debug(
                "Cannot apply restrictive permissions to {}: {}".format(temp_path, exc)
            )
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written : written + _IO_CHUNK_SIZE])
        _fsync_fd(fd, str(temp_path))
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(fd)
    return temp_path


def cleanup_stale_temp_files(config_file: Path) -> None:
    """
    Best effort removal of temporary files left over by a crashed earlier
    transaction in the target directory.
    """
    config_file = Path(config_file)
    directory = config_file.parent
    try:
        candidates = list(
            directory.glob(TEMP_FILE_PREFIX.format(name=config_file.name) + "*")
        )
    except OSError as exc:
        logger.debug(
            "Cannot list stale temporary files in {}: {}".format(directory, exc)
        )
        return
    for candidate in candidates:
        try:
            candidate.unlink()
            logger.info("Removed stale temporary file {}".format(candidate))
        except OSError as exc:
            logger.debug(
                "Cannot remove stale temporary file {}: {}".format(candidate, exc)
            )


def _read_file_bytes(path: Path) -> bytes:
    with open(path, "rb") as file_handle:
        chunks = []
        while True:
            chunk = file_handle.read(_IO_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _read_checksum_sidecar(sidecar_path: Path) -> Optional[str]:
    """
    Parse a sha256sum compatible sidecar ("<hex>  filename") or a bare hex.
    """
    try:
        raw = _read_file_bytes(sidecar_path).decode("utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    token = raw.splitlines()[0].split()[0].strip().lower()
    if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
        return token
    return None


def inspect_generation(path: Path, accept_legacy: bool = False) -> GenerationInfo:
    """
    Inspect a configuration generation at the checksum/bytes level.

    Structural parsing is left to the caller (select_generation), because only
    the caller knows what a valid configuration snapshot looks like.
    """
    path = Path(path)
    if not path.exists():
        return GenerationInfo(path, STATE_ABSENT)

    try:
        data = _read_file_bytes(path)
    except OSError as exc:
        return GenerationInfo(path, STATE_UNREADABLE, detail=str(exc))

    actual_checksum = checksum_sha256(data)
    sidecar = _checksum_sidecar_path(path)
    expected_checksum = _read_checksum_sidecar(sidecar)

    if expected_checksum is None:
        # A versioned generation always comes with a sidecar. A sidecar-less
        # current target is a legacy file from an older npbackup version.
        if accept_legacy:
            return GenerationInfo(
                path,
                STATE_LEGACY,
                checksum_actual=actual_checksum,
                size=len(data),
                data=data,
            )
        return GenerationInfo(
            path,
            STATE_CORRUPT_CHECKSUM,
            checksum_actual=actual_checksum,
            size=len(data),
            detail="missing checksum sidecar {}".format(sidecar),
            data=data,
        )

    if expected_checksum != actual_checksum:
        return GenerationInfo(
            path,
            STATE_CORRUPT_CHECKSUM,
            checksum_expected=expected_checksum,
            checksum_actual=actual_checksum,
            size=len(data),
            detail="checksum mismatch, expected {}".format(expected_checksum),
            data=data,
        )

    return GenerationInfo(
        path,
        STATE_VALID,
        checksum_expected=expected_checksum,
        checksum_actual=actual_checksum,
        size=len(data),
        data=data,
    )


def _publish(config_file: Path, data: bytes, rotate: bool) -> CommitResult:
    """
    Atomically publish data as the current configuration generation.

    When rotate is True (normal commit), the existing current generation is
    moved aside as the single retained previous generation before publishing.

    When rotate is False (recovery from the previous generation), the previous
    generation is left untouched and the (corrupt) current generation is
    replaced by a verified copy of it.
    """
    config_file = Path(config_file)
    directory = config_file.parent
    target_checksum_file = _checksum_sidecar_path(config_file)
    previous_file = _previous_generation_path(config_file)
    previous_checksum_file = _checksum_sidecar_path(previous_file)

    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        message = "Refusing to commit an empty configuration snapshot"
        logger.critical(message)
        return CommitResult(False, config_file=config_file, message=message)
    data = bytes(data)

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        message = "Cannot create configuration directory {}: {}".format(directory, exc)
        logger.critical(message)
        return CommitResult(False, config_file=config_file, message=message)

    cleanup_stale_temp_files(config_file)

    checksum = checksum_sha256(data)
    previous_checksum = None
    if target_checksum_file.exists():
        previous_checksum = _read_checksum_sidecar(target_checksum_file)

    temp_data_path = None
    temp_checksum_path = None
    rotated = False
    try:
        temp_data_path = _write_temp_file(directory, config_file.name, data)

        # Re-read and verify the temporary file before it can become the
        # target: readers must never observe a torn or partial snapshot.
        try:
            written_data = _read_file_bytes(temp_data_path)
        except OSError as exc:
            raise OSError(
                "Cannot verify freshly written temporary file {}: {}".format(
                    temp_data_path, exc
                )
            )
        written_checksum = checksum_sha256(written_data)
        if written_checksum != checksum or written_data != data:
            raise OSError(
                "Temporary file {} verification failed (checksum {} != {})".format(
                    temp_data_path, written_checksum, checksum
                )
            )

        checksum_payload = "{}  {}\n".format(checksum, config_file.name).encode("utf-8")
        temp_checksum_path = _write_temp_file(
            directory, target_checksum_file.name, checksum_payload
        )

        if rotate:
            # Keep exactly one previous generation: drop an older one first.
            for stale_path in (previous_checksum_file, previous_file):
                try:
                    stale_path.unlink()
                except FileNotFoundError:
                    pass
            if config_file.exists():
                os.replace(config_file, previous_file)
                rotated = True
                try:
                    if target_checksum_file.exists():
                        os.replace(target_checksum_file, previous_checksum_file)
                except OSError as exc:
                    logger.warning(
                        "Cannot rotate checksum sidecar {}: {}".format(
                            target_checksum_file, exc
                        )
                    )
        else:
            # Recovery: remove corrupt current generation, keep previous intact.
            for stale_path in (target_checksum_file, config_file):
                try:
                    stale_path.unlink()
                except FileNotFoundError:
                    pass

        os.replace(temp_data_path, config_file)
        temp_data_path = None
        os.replace(temp_checksum_path, target_checksum_file)
        temp_checksum_path = None
        _fsync_directory(directory)
    except OSError as exc:
        message = "Cannot commit configuration file {}: {}".format(config_file, exc)
        logger.critical(message)
        logger.debug("Trace:", exc_info=True)
        # Best effort rollback when the current generation was moved aside but
        # the new one could not be published.
        if rotate and rotated and not config_file.exists():
            try:
                if previous_file.exists():
                    os.replace(previous_file, config_file)
                if previous_checksum_file.exists():
                    os.replace(previous_checksum_file, target_checksum_file)
                _fsync_directory(directory)
                logger.info(
                    "Rolled back configuration file {} from previous generation".format(
                        config_file
                    )
                )
            except OSError as rollback_exc:
                logger.critical(
                    "Cannot rollback configuration file {}: {}".format(
                        config_file, rollback_exc
                    )
                )
        for leftover in (temp_data_path, temp_checksum_path):
            if leftover is not None:
                try:
                    leftover.unlink()
                except OSError:
                    pass
        return CommitResult(
            False,
            config_file=config_file,
            previous_checksum=previous_checksum,
            message=message,
        )

    return CommitResult(
        True,
        config_file=config_file,
        checksum=checksum,
        previous_checksum=previous_checksum,
        bytes_written=len(data),
        recovered=not rotate,
        message="Configuration committed to {}".format(config_file),
    )


def commit_config(config_file: Path, data: bytes) -> CommitResult:
    """
    Transactional commit of a new encrypted configuration snapshot.
    Retains exactly one previous encrypted generation and its checksum.

    Rotation only happens when the current target is itself a valid,
    checksum-verified versioned generation. A legacy (unversioned, possibly
    plaintext) target is replaced in place without being copied to the
    previous generation, so no unversioned plaintext copy is retained.
    """
    config_file = Path(config_file)
    current_info = inspect_generation(config_file, accept_legacy=True)
    rotate = current_info.state == STATE_VALID
    return _publish(config_file, data, rotate=rotate)


def recover_current_from_previous(
    config_file: Path, previous_info: GenerationInfo
) -> CommitResult:
    """
    Re-publish the retained previous generation as the current one without
    destroying the good previous copy.
    """
    if previous_info.data is None:
        return CommitResult(
            False,
            config_file=config_file,
            message="Previous generation {} has no readable data".format(
                previous_info.path
            ),
        )
    result = _publish(config_file, previous_info.data, rotate=False)
    result.recovered = True
    if result.success:
        result.message = (
            "Configuration recovered from previous generation {} "
            "(checksum {})".format(previous_info.path, result.checksum)
        )
    return result


def select_generation(
    config_file: Path, parser: Callable[[bytes], Any]
) -> SelectionResult:
    """
    Select the newest valid configuration generation.

    parser must accept the raw file bytes and return the parsed configuration
    object, or raise an exception when the bytes do not form a valid snapshot.
    """
    config_file = Path(config_file)
    previous_file = _previous_generation_path(config_file)

    current_info = inspect_generation(config_file, accept_legacy=True)
    previous_info = inspect_generation(previous_file, accept_legacy=False)
    generations = [current_info, previous_info]

    def _try_parse(info: GenerationInfo) -> tuple:
        if info.data is None:
            return None, "no data"
        try:
            parsed = parser(info.data)
        except Exception as exc:  # pylint: disable=broad-except
            return None, "{}: {}".format(type(exc).__name__, exc)
        if parsed is None:
            return None, "empty configuration"
        return parsed, ""

    if current_info.state == STATE_VALID:
        parsed, error = _try_parse(current_info)
        if parsed is not None:
            return SelectionResult(
                STATUS_CURRENT,
                data=current_info.data,
                parsed=parsed,
                generations=generations,
                message="Loaded current generation {}".format(config_file),
            )
        current_info.state = STATE_INVALID_FORMAT
        current_info.detail = error
        logger.critical(
            "Current configuration generation {} is checksum-valid but cannot "
            "be parsed: {}".format(config_file, error)
        )
    elif current_info.state == STATE_LEGACY:
        parsed, error = _try_parse(current_info)
        if parsed is not None:
            logger.info(
                "Loaded legacy configuration file {} without versioned "
                "checksum; it will be versioned on next save".format(config_file)
            )
            return SelectionResult(
                STATUS_LEGACY,
                data=current_info.data,
                parsed=parsed,
                generations=generations,
                message="Loaded legacy generation {}".format(config_file),
            )
        current_info.state = STATE_INVALID_FORMAT
        current_info.detail = error
        logger.critical(
            "Legacy configuration file {} cannot be parsed: {}".format(
                config_file, error
            )
        )
    else:
        logger.critical(
            "Current configuration generation {} is unusable: {} ({})".format(
                config_file, current_info.state, current_info.detail
            )
        )

    if previous_info.state == STATE_VALID:
        parsed, error = _try_parse(previous_info)
        if parsed is not None:
            logger.warning(
                "Recovering configuration from previous generation {}".format(
                    previous_file
                )
            )
            recovery = recover_current_from_previous(config_file, previous_info)
            if recovery.success:
                return SelectionResult(
                    STATUS_RECOVERED_PREVIOUS,
                    data=previous_info.data,
                    parsed=parsed,
                    generations=generations,
                    recovery=recovery,
                    message=recovery.message,
                )
            logger.critical(
                "Cannot publish recovered previous generation: {}".format(
                    recovery.message
                )
            )
        else:
            previous_info.state = STATE_INVALID_FORMAT
            previous_info.detail = error
            logger.critical(
                "Previous configuration generation {} is checksum-valid but "
                "cannot be parsed: {}".format(previous_file, error)
            )

    return SelectionResult(
        STATUS_UNAVAILABLE,
        generations=generations,
        message="No valid configuration generation found for {}".format(config_file),
    )
