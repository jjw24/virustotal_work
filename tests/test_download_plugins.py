"""Unit tests for ci/src/download-plugins.py."""

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests

# ---------------------------------------------------------------------------
# Module import: download-plugins.py uses a hyphen, making it non-importable
# by standard import.  Use importlib and add its parent dir to sys.path so
# that ``from _utils import ...`` resolves correctly.
# ---------------------------------------------------------------------------
CI_SRC = Path(__file__).resolve().parent.parent / "ci" / "src"
sys.path.insert(0, str(CI_SRC))

import importlib.util

_MODULE_PATH = CI_SRC / "download-plugins.py"
_spec = importlib.util.spec_from_file_location("download_plugins", str(_MODULE_PATH))
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)


# ===================================================================
# env_int
# ===================================================================


class TestEnvInt:
    def test_returns_default_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("MY_VAR", raising=False)
        assert dp.env_int("MY_VAR", 42) == 42

    def test_returns_default_when_env_empty(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "")
        assert dp.env_int("MY_VAR", 42) == 42

    def test_parses_env_value(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "8")
        assert dp.env_int("MY_VAR", 42) == 8

    def test_default_is_zero(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "0")
        assert dp.env_int("MY_VAR", 1) == 0


# ===================================================================
# manifest_filename
# ===================================================================


class TestManifestFilename:
    def test_returns_correct_filename(self):
        plugin = {"Name": "My Plugin", "ID": "abc-123"}
        assert dp.manifest_filename(plugin) == "My Plugin-abc-123.json"


# ===================================================================
# select_new_plugins
# ===================================================================


class TestSelectNewPlugins:
    def test_returns_only_new_plugins(self):
        all_plugins = [
            {"ID": "1", "Name": "Existing"},
            {"ID": "2", "Name": "New One"},
            {"ID": "3", "Name": "Another New"},
        ]
        with (
            patch.object(dp, "get_new_plugin_submission_ids", return_value=["2", "3"]),
            patch.object(dp, "plugin_reader", return_value=all_plugins),
        ):
            plugins, meta = dp.select_new_plugins()
            assert len(plugins) == 2
            assert plugins[0]["ID"] == "2"
            assert plugins[1]["ID"] == "3"
            assert meta["mode"] == "new"
            assert meta["new_submissions"] == 2

    def test_returns_empty_when_no_new_submissions(self):
        with (
            patch.object(dp, "get_new_plugin_submission_ids", return_value=[]),
            patch.object(dp, "plugin_reader", return_value=[]),
        ):
            plugins, meta = dp.select_new_plugins()
            assert plugins == []
            assert meta["new_submissions"] == 0

    def test_skips_ids_not_in_reader(self):
        all_plugins = [{"ID": "1", "Name": "Only"}]
        with (
            patch.object(dp, "get_new_plugin_submission_ids", return_value=["1", "nonexistent"]),
            patch.object(dp, "plugin_reader", return_value=all_plugins),
        ):
            plugins, _ = dp.select_new_plugins()
            assert len(plugins) == 1
            assert plugins[0]["ID"] == "1"


# ===================================================================
# _github_headers
# ===================================================================


# ===================================================================
# download_plugin
# ===================================================================


class TestDownloadPlugin:
    def test_downloads_successfully(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DOWNLOAD_TIMEOUT_SEC", "30")
        mock_response = MagicMock(spec=requests.Response)
        mock_response.iter_content.return_value = [b"chunk1", b"", b"chunk2"]
        mock_response.__enter__.return_value = mock_response

        with patch.object(dp, "requests") as mock_requests:
            mock_requests.get.return_value = mock_response
            dest = tmp_path / "plugin.zip"
            plugin = {dp.url_download: "https://example.com/plugin.zip"}
            dp.download_plugin(plugin, dest)
            mock_requests.get.assert_called_once_with(
                "https://example.com/plugin.zip",
                timeout=30,
                stream=True,
            )
        assert dest.read_bytes() == b"chunk1chunk2"

    def test_raises_on_http_error(self, tmp_path):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")
        mock_response.__enter__.return_value = mock_response

        with patch.object(dp, "requests") as mock_requests:
            mock_requests.get.return_value = mock_response
            with pytest.raises(requests.HTTPError):
                dp.download_plugin(
                    {dp.url_download: "https://example.com/bad.zip"},
                    tmp_path / "bad.zip",
                )

    def test_raises_on_missing_urldownload(self, tmp_path):
        with pytest.raises(KeyError):
            dp.download_plugin({"Name": "no url"}, tmp_path / "plugin.zip")


# ===================================================================
# sha256_file
# ===================================================================


class TestSha256File:
    def test_computes_correct_hash(self, tmp_path):
        content = b"hello world"
        expected = hashlib.sha256(content).hexdigest()
        f = tmp_path / "data.bin"
        f.write_bytes(content)
        assert dp.sha256_file(f) == expected

    def test_handles_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert dp.sha256_file(f) == hashlib.sha256(b"").hexdigest()


# ===================================================================
# _load_cache_meta
# ===================================================================


class TestLoadCacheMeta:
    def test_loads_existing_file(self, tmp_path):
        data = {"plugin.zip": {"version": "1.0"}}
        f = tmp_path / "cache.json"
        f.write_text(json.dumps(data))
        assert dp._load_cache_meta(f) == data

    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        assert dp._load_cache_meta(tmp_path / "nonexistent.json") == {}


# ===================================================================
# _save_cache_meta
# ===================================================================


class TestSaveCacheMeta:
    def test_saves_to_file(self, tmp_path):
        data = {"plugin.zip": {"version": "1.0"}}
        f = tmp_path / "cache.json"
        dp._save_cache_meta(f, data)
        assert json.loads(f.read_text()) == data

    def test_creates_parent_directories(self, tmp_path):
        data = {"p.zip": {"version": "2.0"}}
        f = tmp_path / "sub" / "nested" / "cache.json"
        dp._save_cache_meta(f, data)
        assert f.exists()
        assert json.loads(f.read_text()) == data


# ===================================================================
# _expected_zip_filenames
# ===================================================================


class TestExpectedZipFilenames:
    def test_returns_correct_filenames(self):
        plugins = [
            {dp.plugin_name: "Plugin A", dp.id_name: "id1"},
            {dp.plugin_name: "Plugin B", dp.id_name: "id2"},
        ]
        result = dp._expected_zip_filenames(plugins)
        assert result == {"Plugin A-id1.zip", "Plugin B-id2.zip"}

    def test_returns_empty_set_for_empty_list(self):
        assert dp._expected_zip_filenames([]) == set()


# ===================================================================
# _prune_orphans
# ===================================================================


class TestPruneOrphans:
    def test_removes_orphan_zips(self, tmp_path):
        (tmp_path / "keep.zip").write_text("keep")
        (tmp_path / "orphan.zip").write_text("orphan")
        (tmp_path / "other.txt").write_text("text")
        cache = {"orphan.zip": {"version": "1"}, "keep.zip": {"version": "2"}}
        dp._prune_orphans(tmp_path, {"keep.zip"}, cache)
        assert (tmp_path / "keep.zip").exists()
        assert not (tmp_path / "orphan.zip").exists()
        assert (tmp_path / "other.txt").exists()
        assert "orphan.zip" not in cache
        assert "keep.zip" in cache

    def test_does_nothing_when_no_orphans(self, tmp_path):
        (tmp_path / "a.zip").write_text("a")
        cache = {"a.zip": {"version": "1"}}
        dp._prune_orphans(tmp_path, {"a.zip"}, cache)
        assert (tmp_path / "a.zip").exists()
        assert cache == {"a.zip": {"version": "1"}}


# ===================================================================
# download_all
# ===================================================================


class TestDownloadAll:
    def make_plugin(self, pid: str, name: str = "", version: str = "1.0") -> dict:
        return {
            dp.id_name: pid,
            dp.plugin_name: name or f"Plugin-{pid}",
            dp.version: version,
            dp.url_download: f"https://example.com/{pid}.zip",
        }

    def test_downloads_all_plugins_fresh(self, tmp_path):
        plugins = [self.make_plugin("1"), self.make_plugin("2")]
        with (
            patch.object(dp, "download_plugin") as mock_dl,
            patch.object(dp, "sha256_file", return_value="abc"),
        ):
            result = dp.download_all(plugins, tmp_path)
        assert len(result) == 2
        for pid, (dest, err, status) in result.items():
            assert err is None
            assert status == "fresh"
        assert mock_dl.call_count == 2

    def test_skips_up_to_date_cached_plugins(self, tmp_path):
        plugin = self.make_plugin("1", version="2.0")
        dest = tmp_path / "Plugin-1-1.zip"
        dest.write_text("existing")
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"Plugin-1-1.zip": {"version": "2.0"}}))
        with patch.object(dp, "download_plugin") as mock_dl:
            result = dp.download_all([plugin], tmp_path, cache_meta_path=cache_path)
        assert result["1"][1] is None  # no error
        assert "up-to-date (v2.0)" in result["1"][2]
        mock_dl.assert_not_called()

    def test_updates_outdated_cached_plugins(self, tmp_path):
        plugin = self.make_plugin("1", version="2.0")
        dest = tmp_path / "Plugin-1-1.zip"
        dest.write_text("old")
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"Plugin-1-1.zip": {"version": "1.0"}}))
        with patch.object(dp, "download_plugin") as mock_dl:
            result = dp.download_all([plugin], tmp_path, cache_meta_path=cache_path)
        assert result["1"][1] is None
        assert "updated (v1.0 -> v2.0)" in result["1"][2]
        mock_dl.assert_called_once()

    def test_handles_download_failures(self, tmp_path):
        plugin = self.make_plugin("fail")
        with patch.object(dp, "download_plugin", side_effect=ValueError("bad")):
            result = dp.download_all([plugin], tmp_path)
        pid, (dest, err, status) = next(iter(result.items()))
        assert err is not None
        assert "ValueError: bad" in err
        assert status is None

    def test_missing_urldownload_in_task(self, tmp_path):
        plugin = {dp.id_name: "1", dp.plugin_name: "P1", dp.version: "1.0"}
        with patch.object(dp, "download_plugin"):
            result = dp.download_all([plugin], tmp_path)
        _, (dest, err, status) = next(iter(result.items()))
        assert err is not None
        assert "missing UrlDownload" in err
        assert status is None

    def test_http_error_with_response(self, tmp_path):
        plugin = self.make_plugin("1")
        resp = MagicMock()
        resp.status_code = 404
        http_err = requests.HTTPError("Not Found")
        http_err.response = resp
        with patch.object(dp, "download_plugin", side_effect=http_err):
            result = dp.download_all([plugin], tmp_path)
        _, (dest, err, status) = next(iter(result.items()))
        assert err is not None
        assert "HTTP 404" in err
        assert status is None

    def test_prunes_orphans(self, tmp_path):
        (tmp_path / "orphan.zip").write_text("orphan")
        plugin = self.make_plugin("1")
        with (
            patch.object(dp, "download_plugin"),
            patch.object(dp, "sha256_file", return_value="abc"),
        ):
            dp.download_all([plugin], tmp_path)
        assert not (tmp_path / "orphan.zip").exists()

    def test_persists_cache_meta(self, tmp_path):
        plugin = self.make_plugin("1", version="3.0")
        cache_path = tmp_path / "cache.json"
        with (
            patch.object(dp, "download_plugin"),
            patch.object(dp, "sha256_file", return_value="abc"),
        ):
            dp.download_all([plugin], tmp_path / "out", cache_meta_path=cache_path)
        assert cache_path.exists()
        meta = json.loads(cache_path.read_text())
        assert "Plugin-1-1.zip" in meta
        assert meta["Plugin-1-1.zip"]["version"] == "3.0"

    def test_mixed_success_and_failure(self, tmp_path):
        plugins = [self.make_plugin("1"), self.make_plugin("2")]
        side_effects = [None, ValueError("fail")]
        with (
            patch.object(dp, "download_plugin", side_effect=side_effects),
            patch.object(dp, "sha256_file", return_value="abc"),
        ):
            result = dp.download_all(plugins, tmp_path)
        assert result["1"][1] is None
        assert result["2"][1] is not None


# ===================================================================
# main (CLI entry point)
# ===================================================================


class TestMainCli:
    def make_plugin(self, pid: str, name: str = "", version: str = "1.0") -> dict:
        return {
            dp.id_name: pid,
            dp.plugin_name: name or f"Plugin-{pid}",
            dp.version: version,
            dp.url_download: f"https://example.com/{pid}.zip",
        }

    def test_mode_all(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["download-plugins.py"])
        with (
            patch.object(dp, "download_all") as mock_download_all,
            patch.object(dp, "plugin_reader", return_value=[self.make_plugin("1")]),
        ):
            dp.main()
        mock_download_all.assert_called_once()

    def test_mode_new(self, monkeypatch):
        monkeypatch.setenv("MODE", "new")
        monkeypatch.setattr(sys, "argv", ["download-plugins.py"])
        with (
            patch.object(dp, "download_all") as mock_download_all,
            patch.object(dp, "select_new_plugins") as mock_select,
        ):
            mock_select.return_value = (
                [self.make_plugin("new1")],
                {"mode": "new", "new_submissions": 1},
            )
            dp.main()
        mock_download_all.assert_called_once()

    def test_mode_new_via_cli_arg(self, monkeypatch):
        monkeypatch.delenv("MODE", raising=False)
        monkeypatch.setattr(sys, "argv", ["download-plugins.py", "--mode", "new"])
        with (
            patch.object(dp, "download_all") as mock_download_all,
            patch.object(dp, "select_new_plugins") as mock_select,
        ):
            mock_select.return_value = (
                [self.make_plugin("cli-new")],
                {"mode": "new", "new_submissions": 1},
            )
            dp.main()
        mock_download_all.assert_called_once()

    def test_exits_when_no_plugins_and_all_mode(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["download-plugins.py"])
        with (
            patch.object(dp, "plugin_reader", return_value=[]),
            pytest.raises(SystemExit) as exc,
        ):
            dp.main()
        assert exc.value.code == 0
