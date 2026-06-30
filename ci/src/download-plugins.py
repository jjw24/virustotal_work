"""Download Flow Launcher plugin ZIPs from their GitHub release URLs.

This script reads plugin manifest JSON files from the ``plugins/``
directory and downloads each plugin's ``UrlDownload`` ZIP into an output
directory.  It supports three selection modes, optional batch processing,
and a local metadata cache to avoid re-downloading unchanged versions.

Selection modes (``--mode``):
    batch       Download all plugins, optionally split across multiple
                CI workflow runs using ``--batch-count`` /
                ``--batch-index`` or ``--plugins-per-batch`` (default).
    new         Download only plugins whose IDs are not yet in the
                published ``plugins.json`` index.
    plugins     Download one or more specific plugins listed by their
                manifest filename (``--plugins Name-ID.json,...``).

Usage examples:

    # Download all plugins (default batch mode, single batch)
    python ci/src/download-plugins.py

    # Download 1 of 4 batches.  Each day picks a different batch
    # (UTC day % 4), so all plugins are covered over 4 days.
    python ci/src/download-plugins.py --batch-count 4

    # Download only newly submitted plugins
    python ci/src/download-plugins.py --mode new

    # Download one specific plugin
    python ci/src/download-plugins.py --plugins FooPlugin-abc123.json

    # Download plugins 100-199 out of the full sorted list
    python ci/src/download-plugins.py --start 100 --count 100

    # Use a cache metadata file to skip unchanged downloads
    python ci/src/download-plugins.py --cache-meta cache.json

Environment variables:
    GITHUB_TOKEN         Required.  GitHub PAT with ``repo`` scope.
    MODE                 Fallback for ``--mode``.
    PLUGINS              Fallback for ``--plugins``.
    BATCH_COUNT          Fallback for ``--batch-count``.
    BATCH_INDEX          Fallback for ``--batch-index``.
    PLUGINS_PER_BATCH    Fallback for ``--plugins-per-batch``.
    OUTPUT_DIR           Fallback for ``--output-dir`` (default: plugin_downloads).
    DOWNLOAD_WORKERS     Max concurrent downloads (default: 8).
    DOWNLOAD_TIMEOUT_SEC HTTP request timeout in seconds (default: 120).
"""

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
from _utils import get_new_plugin_submission_ids, id_name, plugin_dir, plugin_name, url_download, version

DOWNLOAD_WORKERS = 8


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable.

    Args:
        name: Name of the environment variable.
        default: Value to return if the variable is unset or empty.

    Returns:
        The integer value of the environment variable, or the default.
    """
    val = os.getenv(name, "")
    if not val:
        return default
    return int(val)


# ---------------------------------------------------------------------------
# Plugin utilities
# ---------------------------------------------------------------------------


def manifest_filename(plugin: dict[str, str]) -> str:
    """Build the manifest filename for a plugin.

    Args:
        plugin: A plugin dictionary containing ``Name`` and ``ID`` keys.

    Returns:
        A string in the format ``{Name}-{ID}.json``.
    """
    return f"{plugin[plugin_name]}-{plugin[id_name]}.json"


def stable_sort_plugins(plugins: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return plugins sorted by (name, ID) case-insensitively.

    Args:
        plugins: List of plugin dictionaries.

    Returns:
        A new sorted list of plugins.
    """
    return sorted(plugins, key=lambda p: (p.get(plugin_name, "").lower(), p.get(id_name, "").lower()))


def parse_plugin_tokens(raw: str) -> list[str]:
    """Split a comma-or-newline-separated string into stripped tokens.

    Args:
        raw: Raw comma/newline-separated input.

    Returns:
        List of non-empty stripped tokens.
    """
    tokens = []
    for part in raw.replace("\n", ",").split(","):
        part = part.strip()
        if part:
            tokens.append(part)
    return tokens


def load_plugins_by_manifest_names(filenames: list[str]) -> list[dict[str, str]]:
    """Load plugin dicts from manifest JSON files by filename.

    Args:
        filenames: List of ``.json`` filenames in the plugin directory.

    Raises:
        SystemExit: If any filename does not end in ``.json`` or its
            corresponding file does not exist.

    Returns:
        List of parsed plugin dictionaries.
    """
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
    """Abstract strategy for selecting which plugins to download."""

    @abstractmethod
    def select(self) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Return the selected plugins and associated metadata.

        Returns:
            A tuple of ``(plugins, metadata_dict)``.
        """
        ...


def resolve_batch(
    plugins: list[dict[str, str]],
    *,
    batch_count: Optional[int] = None,
    batch_index: Optional[int] = None,
    plugins_per_batch: Optional[int] = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Partition a sorted plugin list into batches and return one batch.

    When ``plugins_per_batch`` is provided the total number of batches is
    derived from it; otherwise ``batch_count`` is used (default 1).
    When ``batch_index`` is omitted the current UTC day number is used.

    Args:
        plugins: Full list of plugin dictionaries (will be sorted internally).
        batch_count: Desired number of batches (ignored if
            ``plugins_per_batch`` is set).
        batch_index: Index of the batch to return.  ``None`` uses
            ``(UTC_day_number % batch_count)`` so each day picks a
            different batch, cycling through all batches over time.
        plugins_per_batch: Fixed number of plugins per batch.

    Returns:
        Tuple of ``(selected_plugins, metadata_dict)`` where metadata
        includes keys such as ``total_plugins``, ``batch_count``,
        ``batch_index``, ``batch_size``, etc.
    """
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
    """Select all plugins, optionally sub-divided into batches."""

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
        """Load all manifests and resolve the current batch.

        Returns:
            Tuple of ``(plugins, metadata_dict)``.
        """
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
    """Select only newly-submitted plugins (not yet in ``plugins.json``)."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def select(self) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Select plugins whose IDs are absent from the published index.

        Returns:
            Tuple of ``(new_plugins, metadata_dict)``.
        """
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
    """Select plugins by explicit manifest filenames."""

    def __init__(self, raw: str = "", **_kwargs: Any) -> None:
        self._raw = raw

    def select(self) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Parse the raw token string and load the corresponding manifests.

        Raises:
            SystemExit: If the token string is empty.

        Returns:
            Tuple of ``(plugins, metadata_dict)``.
        """
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
    """Registry mapping mode names to ``PluginSelector`` implementations."""

    _selectors: dict[str, type[PluginSelector]] = {}

    @classmethod
    def register(cls, mode: str) -> Any:
        """Decorator that registers a selector class for the given mode.

        Args:
            mode: Mode name (e.g. ``"batch"``, ``"new"``, ``"plugins"``).

        Returns:
            A decorator that registers the class and returns it unchanged.
        """

        def decorator(selector_cls: type[PluginSelector]) -> type[PluginSelector]:
            cls._selectors[mode] = selector_cls
            return selector_cls

        return decorator

    @classmethod
    def create(cls, mode: str, **kwargs: Any) -> PluginSelector:
        """Factory: instantiate the selector registered for *mode*.

        Args:
            mode: Mode name.
            **kwargs: Forwarded to the selector's constructor.

        Raises:
            SystemExit: If *mode* is not registered.

        Returns:
            A ``PluginSelector`` instance.
        """
        selector_cls = cls._selectors.get(mode)
        if selector_cls is None:
            raise SystemExit(f"Unknown mode: {mode}")
        return selector_cls(**kwargs)

    @classmethod
    def available_modes(cls) -> list[str]:
        """List all registered mode names.

        Returns:
            Sorted list of mode strings.
        """
        return list(cls._selectors.keys())


PluginSelectorRegistry.register("batch")(BatchPluginSelector)
PluginSelectorRegistry.register("new")(NewPluginSelector)
PluginSelectorRegistry.register("plugins")(SpecificPluginSelector)


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------


def _github_headers() -> dict[str, str]:
    """Build authorization headers for GitHub API requests.

    Returns:
        A dict with an ``Authorization`` header set to a ``token``-type
        GitHub PAT, or an empty dict if ``GITHUB_TOKEN`` is not set.
    """
    token = os.getenv("GITHUB_TOKEN", "")
    return {"Authorization": f"token {token}"}


def download_plugin(plugin: dict[str, str], dest: Path) -> None:
    """Download a plugin ZIP to *dest*.

    Args:
        plugin: Plugin dictionary containing ``UrlDownload``.
        dest: Local path where the ZIP is saved.

    Raises:
        requests.HTTPError: On non-2xx HTTP responses.
    """
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
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        Lower-case hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _load_cache_meta(path: Path) -> dict[str, Any]:
    """Load cache metadata from a JSON file.

    Args:
        path: Path to the cache metadata file.

    Returns:
        Deserialised dictionary, or an empty dict if the file does not exist.
    """
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cache_meta(path: Path, meta: dict[str, Any]) -> None:
    """Persist cache metadata to a JSON file.

    Args:
        path: Destination path.
        meta: Metadata dictionary to serialise.
    """
    first_time = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    if first_time:
        print(f"Plugin download cache metadata created at {path}")


def _expected_zip_filenames(plugins: list[dict[str, str]]) -> set[str]:
    """Compute the set of expected ZIP filenames for the given plugins.

    Args:
        plugins: List of plugin dictionaries.

    Returns:
        Set of ``{Name}-{ID}.zip`` strings.
    """
    return {manifest_filename(p).replace(".json", "") + ".zip" for p in plugins}


def _prune_orphans(output_dir: Path, expected_filenames: set[str], cache_meta: dict[str, Any]) -> None:
    """Remove ZIP files in *output_dir* not in *expected_filenames*.

    Also removes the corresponding entries from *cache_meta*.

    Args:
        output_dir: Directory containing downloaded ZIP files.
        expected_filenames: Set of filenames that should be kept.
        cache_meta: In-memory cache metadata dict (mutated in-place).
    """
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
    """Download all plugin ZIPs in parallel.

    Skips plugins whose cached version matches the manifest version.
    Prunes orphan ZIPs and updates the cache metadata.

    Args:
        plugins: List of plugin dictionaries to download.
        output_dir: Directory to write ZIP files into.
        cache_meta_path: Optional path to a JSON cache metadata file.

    Returns:
        Dict mapping plugin ID to ``(dest_path, error_or_None)``.
    """
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
            print(f"  FAIL {dest.name}: {err}")
        elif status and status.startswith("up-to-date"):
            print(f"  From cache -> {dest.name}: {status}")
        elif status and status.startswith("updated"):
            print(f"  From cache -> {dest.name}: {status}")
        else:
            sha = sha256_file(dest)
            print(f"  Downloaded {dest.name} sha256={sha[:12]}...")
    return out


# ---------------------------------------------------------------------------
# Batch config resolver
# ---------------------------------------------------------------------------


def resolve_batch_config(args: argparse.Namespace) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Resolve batch parameters from CLI args and environment variables.

    Priority: CLI argument > environment variable.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of ``(batch_count, batch_index, plugins_per_batch)``.
    """
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
    """Parse CLI arguments and orchestrate the download workflow."""
    parser = argparse.ArgumentParser(description="Download Flow Launcher plugin zips")
    parser.add_argument(
        "--mode",
        default=None,
        choices=PluginSelectorRegistry.available_modes(),
        help="Selection mode (falls back to MODE env var, defaults to batch)",
    )
    parser.add_argument(
        "--plugins", default=None, help="Comma-separated manifest filenames (falls back to PLUGINS env var)"
    )
    parser.add_argument(
        "--batch-count", type=int, default=None, help="Number of batches (falls back to BATCH_COUNT env var)"
    )
    parser.add_argument(
        "--batch-index", type=int, default=None, help="Current batch index (falls back to BATCH_INDEX env var)"
    )
    parser.add_argument(
        "--plugins-per-batch",
        type=int,
        default=None,
        help="Plugins per batch (falls back to PLUGINS_PER_BATCH env var)",
    )
    parser.add_argument("--start", type=int, default=0, help="0-based index of first plugin to download (default: 0)")
    parser.add_argument(
        "--count", type=int, default=0, help="Number of plugins to download (0 = all remaining from --start)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: plugin_downloads, falls back to OUTPUT_DIR env var)",
    )
    parser.add_argument(
        "--cache-meta",
        default=None,
        help="Path to cache metadata JSON; skips download if cached version matches manifest",
    )
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
    selector = PluginSelectorRegistry.create(
        args.mode, batch_count=bc, batch_index=bi, plugins_per_batch=ppb, raw=args.plugins
    )
    plugins, meta = selector.select()
    if not plugins:
        print("No plugins to download")
        sys.exit(0)

    start = args.start
    count = args.count
    if start > 0 or count > 0:
        if count > 0:
            plugins = plugins[start : start + count]
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
