"""Download plugins from Flow Launcher plugin manifest files."""

import argparse
import hashlib
import json
import math
import os
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import requests

from _utils import (
    get_new_plugin_submission_ids,
    id_name,
    plugin_dir,
    plugin_name,
    url_download,
    version,
)

DOWNLOAD_WORKERS = 8


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def env_int(name: str, default: int) -> int:
    val = os.getenv(name, "")
    if not val:
        return default
    return int(val)


# ---------------------------------------------------------------------------
# Plugin utilities
# ---------------------------------------------------------------------------

def manifest_filename(plugin: dict[str, str]) -> str:
    return f"{plugin[plugin_name]}-{plugin[id_name]}.json"


def stable_sort_plugins(plugins: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(plugins, key=lambda p: (p.get(plugin_name, "").lower(), p.get(id_name, "").lower()))


def parse_plugin_tokens(raw: str) -> list[str]:
    tokens = []
    for part in raw.replace("\n", ",").split(","):
        part = part.strip()
        if part:
            tokens.append(part)
    return tokens


def load_plugins_by_manifest_names(filenames: list[str]) -> list[dict[str, str]]:
    plugins = []
    missing = []
    for token in filenames:
        if not token.endswith(".json"):
            missing.append(token)
            continue
        path = plugin_dir / token
        if not path.is_file():
            missing.append(token)
            continue
        with open(path, "r", encoding="utf-8") as f:
            plugins.append(json.load(f))
    if missing:
        raise SystemExit(f"MANIFEST_ERROR: unknown manifest file(s): {', '.join(missing)}")
    return plugins


# ---------------------------------------------------------------------------
# PluginSelector strategy
# ---------------------------------------------------------------------------

class PluginSelector(ABC):
    @abstractmethod
    def select(self) -> tuple[list[dict[str, str]], dict[str, Any]]:
        ...


def resolve_batch(
    plugins: list[dict[str, str]],
    *,
    batch_count: Optional[int] = None,
    batch_index: Optional[int] = None,
    plugins_per_batch: Optional[int] = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    sorted_plugins = stable_sort_plugins(plugins)
    total = len(sorted_plugins)
    meta: dict[str, Any] = {"total_plugins": total}

    if plugins_per_batch is not None and plugins_per_batch > 0:
        per = plugins_per_batch
        n_batches = max(1, math.ceil(total / per))
        meta["batch_count"] = n_batches
        meta["plugins_per_batch"] = per
    else:
        n_batches = batch_count if batch_count and batch_count > 0 else 1
        per = math.ceil(total / n_batches) if total else 0
        meta["batch_count"] = n_batches

    if n_batches <= 1:
        meta["batch_index"] = 0
        meta["batch_size"] = total
        return sorted_plugins, meta

    if batch_index is None:
        import time as _time
        import datetime as _datetime
        from datetime import timezone as _timezone
        days = int(_datetime.datetime.now(_timezone.utc).timestamp() // 86400)
        batch_index = days % n_batches
    meta["batch_index"] = batch_index

    start = batch_index * per
    end = min(start + per, total)
    slice_plugins = sorted_plugins[start:end]
    meta["batch_start"] = start
    meta["batch_end"] = end
    meta["batch_size"] = len(slice_plugins)
    return slice_plugins, meta


class BatchPluginSelector(PluginSelector):
    def __init__(
        self,
        batch_count: Optional[int] = None,
        batch_index: Optional[int] = None,
        plugins_per_batch: Optional[int] = None,
        **_kwargs: Any,
    ) -> None:
        self._batch_count = batch_count
        self._batch_index = batch_index
        self._plugins_per_batch = plugins_per_batch

    def select(self) -> tuple[list[dict[str, str]], dict[str, Any]]:
        from _utils import plugin_reader

        all_plugins = plugin_reader()
        plugins, batch_meta = resolve_batch(
            all_plugins,
            batch_count=self._batch_count,
            batch_index=self._batch_index,
            plugins_per_batch=self._plugins_per_batch,
        )
        meta: dict[str, Any] = {"mode": "batch"}
        meta.update(batch_meta)

        if meta.get("batch_count", 1) > 1:
            print(
                f"Batch {meta.get('batch_index', 0) + 1}/{meta['batch_count']}: "
                f"plugins {meta.get('batch_start', 0)}-"
                f"{meta.get('batch_end', 0) - 1} of {meta['total_plugins']}"
            )
        else:
            print(f"Downloading all {len(plugins)} plugins")
        return plugins, meta


class NewPluginSelector(PluginSelector):
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def select(self) -> tuple[list[dict[str, str]], dict[str, Any]]:
        ids = get_new_plugin_submission_ids()
        from _utils import plugin_reader
        by_id = {p[id_name]: p for p in plugin_reader()}
        plugins = [by_id[i] for i in ids if i in by_id]
        meta: dict[str, Any] = {"mode": "new", "new_submissions": len(plugins)}
        if not plugins:
            print("No new plugin submissions to download")
        else:
            print(f"Downloading {len(plugins)} new plugin submission(s)")
        return plugins, meta


class SpecificPluginSelector(PluginSelector):
    def __init__(self, raw: str = "", **_kwargs: Any) -> None:
        self._raw = raw

    def select(self) -> tuple[list[dict[str, str]], dict[str, Any]]:
        if not self._raw.strip():
            raise SystemExit("MANIFEST_ERROR: --plugins or PLUGINS env var required for mode=plugins")
        plugins = load_plugins_by_manifest_names(parse_plugin_tokens(self._raw))
        meta: dict[str, Any] = {
            "mode": "plugins",
            "manifest_files": [manifest_filename(p) for p in plugins],
        }
        print(f"Downloading {len(plugins)} selected plugin(s)")
        return plugins, meta


class PluginSelectorRegistry:
    _selectors: dict[str, type[PluginSelector]] = {}

    @classmethod
    def register(cls, mode: str) -> Any:
        def decorator(selector_cls: type[PluginSelector]) -> type[PluginSelector]:
            cls._selectors[mode] = selector_cls
            return selector_cls
        return decorator

    @classmethod
    def create(cls, mode: str, **kwargs: Any) -> PluginSelector:
        selector_cls = cls._selectors.get(mode)
        if selector_cls is None:
            raise SystemExit(f"Unknown mode: {mode}")
        return selector_cls(**kwargs)

    @classmethod
    def available_modes(cls) -> list[str]:
        return list(cls._selectors.keys())


PluginSelectorRegistry.register("batch")(BatchPluginSelector)
PluginSelectorRegistry.register("new")(NewPluginSelector)
PluginSelectorRegistry.register("plugins")(SpecificPluginSelector)


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def _github_headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN", "")
    return {"Authorization": f"token {token}"}


def download_plugin(plugin: dict[str, str], dest: Path) -> None:
    url = plugin[url_download]

    timeout = env_int("DOWNLOAD_TIMEOUT_SEC", 120)
    headers = _github_headers()
    with requests.get(url, headers=headers, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache_meta(path: Path) -> dict[str, Any]:
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cache_meta(path: Path, meta: dict[str, Any]) -> None:
    first_time = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    if first_time:
        print(f"Plugin download cache metadata created at {path}")


def _expected_zip_filenames(plugins: list[dict[str, str]]) -> set[str]:
    return {manifest_filename(p).replace(".json", "") + ".zip" for p in plugins}


def _prune_orphans(output_dir: Path, expected_filenames: set[str],
                   cache_meta: dict[str, Any]) -> None:
    pruned = []
    for f in list(output_dir.glob("*.zip")):
        if f.name not in expected_filenames:
            f.unlink()
            cache_meta.pop(f.name, None)
            pruned.append(f.name)
    if pruned:
        print(f"Pruned {len(pruned)} orphaned plugin ZIP(s):")
        for name in sorted(pruned):
            print(f"  removed {name}")


# ---------------------------------------------------------------------------
# Download step
# ---------------------------------------------------------------------------

def download_all(
    plugins: list[dict[str, str]],
    output_dir: Path,
    cache_meta_path: Optional[Path] = None,
) -> dict[str, tuple[Path, Optional[str]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = _load_cache_meta(cache_meta_path) if cache_meta_path else {}
    workers = env_int("DOWNLOAD_WORKERS", DOWNLOAD_WORKERS)
    out: dict[str, tuple[Path, Optional[str]]] = {}

    def task(plugin: dict[str, str]) -> tuple[str, Path, Optional[str], Optional[str]]:
        pid = plugin[id_name]
        dest = output_dir / f"{manifest_filename(plugin).replace('.json', '')}.zip"
        filename = dest.name
        cached = cache_meta.get(filename)
        if cached and cached.get("version") == plugin.get(version) and dest.exists():
            return pid, dest, None, f"up-to-date (v{cached['version']})"
        try:
            if not plugin.get(url_download):
                raise ValueError("missing UrlDownload")
            download_plugin(plugin, dest)
            cache_meta[filename] = {"version": plugin.get(version, "")}
            status = f"updated (v{cached['version']} -> v{plugin.get(version, '')})" if cached else "fresh"
            return pid, dest, None, status
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if isinstance(e, requests.HTTPError) and e.response is not None:
                err = f"HTTP {e.response.status_code} {plugin.get(url_download, '')} {e}"
            return pid, dest, err, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, p): p for p in plugins}
        for fut in as_completed(futures):
            pid, dest, err, status = fut.result()
            out[pid] = (dest, err, status)

    expected_filenames = _expected_zip_filenames(plugins)
    _prune_orphans(output_dir, expected_filenames, cache_meta)

    if cache_meta_path:
        _save_cache_meta(cache_meta_path, cache_meta)

    total = len(plugins)
    ok = sum(1 for v in out.values() if v[1] is None)
    failed = total - ok
    print(f"\nProcessed {ok}/{total} plugins" + (f" ({failed} failed)" if failed else ""))
    for pid, (dest, err, status) in out.items():
        if err:
            print(f"  FAIL {pid}: {err}")
        elif status and status.startswith("up-to-date"):
            print(f"  Cached {pid}: {status}")
        elif status and status.startswith("updated"):
            print(f"  Cached {pid}: {status}")
        else:
            sha = sha256_file(dest)
            print(f"  Downloaded {pid} -> {dest.name} sha256={sha[:12]}...")
    return out


# ---------------------------------------------------------------------------
# Batch config resolver
# ---------------------------------------------------------------------------

def resolve_batch_config(args: argparse.Namespace) -> tuple[Optional[int], Optional[int], Optional[int]]:
    batch_count = args.batch_count
    if batch_count is None and os.getenv("BATCH_COUNT"):
        batch_count = int(os.getenv("BATCH_COUNT", "1"))
    batch_index = args.batch_index
    if batch_index is None and os.getenv("BATCH_INDEX") != "":
        val = os.getenv("BATCH_INDEX")
        if val is not None and val != "":
            batch_index = int(val)
    plugins_per_batch = args.plugins_per_batch
    if plugins_per_batch is None and os.getenv("PLUGINS_PER_BATCH"):
        plugins_per_batch = int(os.getenv("PLUGINS_PER_BATCH", "0")) or None
    return batch_count, batch_index, plugins_per_batch


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download Flow Launcher plugin zips")
    parser.add_argument("--mode", default=None, choices=PluginSelectorRegistry.available_modes(),
                        help="Selection mode (falls back to MODE env var, defaults to batch)")
    parser.add_argument("--plugins", default=None,
                        help="Comma-separated manifest filenames (falls back to PLUGINS env var)")
    parser.add_argument("--batch-count", type=int, default=None,
                        help="Number of batches (falls back to BATCH_COUNT env var)")
    parser.add_argument("--batch-index", type=int, default=None,
                        help="Current batch index (falls back to BATCH_INDEX env var)")
    parser.add_argument("--plugins-per-batch", type=int, default=None,
                        help="Plugins per batch (falls back to PLUGINS_PER_BATCH env var)")
    parser.add_argument("--start", type=int, default=0,
                        help="0-based index of first plugin to download (default: 0)")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of plugins to download (0 = all remaining from --start)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: plugin_downloads, falls back to OUTPUT_DIR env var)")
    parser.add_argument("--cache-meta", default=None,
                        help="Path to cache metadata JSON; skips download if cached version matches manifest")
    args = parser.parse_args()

    if args.plugins is None:
        args.plugins = os.getenv("PLUGINS", "")

    if args.plugins.strip():
        args.mode = "plugins"
    elif args.mode is None:
        args.mode = os.getenv("MODE")
    if args.mode is None:
        args.mode = "batch"

    if not os.getenv("GITHUB_TOKEN"):
        print("GITHUB_TOKEN is required- set it to a GitHub PAT with repo scope")
        sys.exit(1)

    output_dir = Path(args.output_dir or os.getenv("OUTPUT_DIR", "plugin_downloads"))

    bc, bi, ppb = resolve_batch_config(args)
    selector = PluginSelectorRegistry.create(args.mode, batch_count=bc, batch_index=bi,
                                              plugins_per_batch=ppb, raw=args.plugins)
    plugins, meta = selector.select()
    if not plugins:
        print("No plugins to download")
        sys.exit(0)

    start = args.start
    count = args.count
    if start > 0 or count > 0:
        if count > 0:
            plugins = plugins[start:start + count]
        else:
            plugins = plugins[start:]
        meta["slice_start"] = start
        meta["slice_count"] = len(plugins)

    if not plugins:
        print("No plugins to download")
        sys.exit(0)

    if start > 0 or count > 0:
        print(f"Downloading plugins {start}-{start + len(plugins) - 1} of {meta.get('total_plugins', '?')}")

    cache_meta_path = Path(args.cache_meta) if args.cache_meta else None
    download_all(plugins, output_dir, cache_meta_path=cache_meta_path)


if __name__ == "__main__":
    main()
