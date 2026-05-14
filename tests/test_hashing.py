"""Tests para `churnlens.utils.hashing`."""

from __future__ import annotations

from pathlib import Path

from churnlens.utils.hashing import (
    build_checksum_record,
    compute_md5,
    compute_sha256,
    load_checksums,
    save_checksums,
    verify_md5,
)


def _write(tmp: Path, content: bytes) -> Path:
    p = tmp / "sample.bin"
    p.write_bytes(content)
    return p


def test_md5_known_value(tmp_path: Path) -> None:
    """MD5 de 'hello world' es conocido públicamente."""
    p = _write(tmp_path, b"hello world")
    assert compute_md5(p) == "5eb63bbbe01eeed093cb22bb8f5acdc3"


def test_sha256_known_value(tmp_path: Path) -> None:
    p = _write(tmp_path, b"hello world")
    assert compute_sha256(p) == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_verify_md5_match(tmp_path: Path) -> None:
    p = _write(tmp_path, b"foo")
    assert verify_md5(p, compute_md5(p)) is True


def test_verify_md5_mismatch(tmp_path: Path) -> None:
    p = _write(tmp_path, b"foo")
    assert verify_md5(p, "0" * 32) is False


def test_verify_md5_empty_expected_is_ok(tmp_path: Path) -> None:
    """Un expected vacío debe pasar (= verificación desactivada)."""
    p = _write(tmp_path, b"foo")
    assert verify_md5(p, "") is True


def test_record_and_persist_checksums(tmp_path: Path) -> None:
    p = _write(tmp_path, b"data")
    record = build_checksum_record(p)
    assert set(record.keys()) >= {"md5", "sha256", "bytes", "downloaded_at"}
    assert record["bytes"] == 4

    out = tmp_path / "checksums.json"
    save_checksums({p.name: record}, out)
    loaded = load_checksums(out)
    assert loaded[p.name]["md5"] == record["md5"]


def test_load_checksums_missing_file(tmp_path: Path) -> None:
    assert load_checksums(tmp_path / "does_not_exist.json") == {}
