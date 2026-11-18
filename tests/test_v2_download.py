"""Acquisition must never publish an HTML response or a mismatched archive."""

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("v2_download", Path("scripts/v2/download_source.py"))
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


@pytest.mark.parametrize("valid", [True, False])
def test_download_checksum_and_immutable_reuse(tmp_path, monkeypatch, valid):
    if not shutil.which("curl"):
        pytest.skip("curl is required by the DGX downloader")
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("example.txt", "synthetic acquisition fixture")
    expected = hashlib.sha256(archive.read_bytes()).hexdigest() if valid else "0" * 64
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/v2_sources.json").write_text(
        json.dumps(
            {
                "iqon3000": {
                    "url": archive.as_uri(),
                    "expected_sha256": expected,
                    "source_page": "synthetic fixture",
                    "relative_path": "iqon/source.zip",
                }
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REPBENCH_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(download, "require_budget", lambda *args: 9999)
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["download", "--source", "iqon3000", "--report", str(report)])
    with pytest.raises(SystemExit) as exc:
        download.main()
    assert exc.value.code == (0 if valid else 1)
    assert (tmp_path / "data/iqon/source.zip").exists() == valid
    assert json.loads(report.read_text())["status"] == ("PASS" if valid else "FAIL")
    with pytest.raises(FileExistsError):
        download.main()
    if valid:
        monkeypatch.setattr(
            sys,
            "argv",
            ["download", "--source", "iqon3000", "--report", str(tmp_path / "reuse.json")],
        )
        with pytest.raises(SystemExit) as exc:
            download.main()
        assert exc.value.code == 0
        assert json.loads((tmp_path / "reuse.json").read_text())["reused"]
