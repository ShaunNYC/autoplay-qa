"""Content identity for the actual Unity player used by a controlled study."""

import hashlib
import json
from pathlib import Path
import stat


def fingerprint_unity_player(player_path):
    """Hash the entry point and complete app/player data, without exposing paths.

    macOS apps include the whole bundle. Other standalone layouts include the
    executable, its adjacent *_Data tree and common Unity runtime payloads.
    OS libraries outside the build are not claimed to be pinned. External
    symlinks and files changing during this read fail closed.
    """
    entry = Path(player_path).expanduser().absolute()
    if not entry.is_file():
        raise ValueError("The configured Unity player is not a file.")
    bundle = next((parent for parent in entry.parents if parent.suffix == ".app"), None)
    if bundle is not None:
        root = bundle.resolve()
        if not entry.resolve().is_relative_to(root):
            raise ValueError("The Unity entry point resolves outside its app bundle.")
        roots = [bundle]
        layout = "macos_app_bundle"
        relative_entry = entry.relative_to(bundle).as_posix()
    else:
        root = entry.parent.resolve()
        roots = [entry]
        for name in (entry.stem + "_Data", "UnityPlayer.so", "UnityPlayer.dll",
                     "GameAssembly.so", "GameAssembly.dll", "MonoBleedingEdge"):
            candidate = entry.parent / name
            if candidate.exists() or candidate.is_symlink():
                roots.append(candidate)
        layout = "standalone_player_payload"
        relative_entry = entry.name

    def entries():
        found = {}
        for selected in roots:
            paths = [selected]
            if selected.is_dir() and not selected.is_symlink():
                paths.extend(selected.rglob("*"))
            for path in paths:
                # Keep lexical names for link identity; all targets must stay
                # inside the same build root whose real files are also hashed.
                base = bundle if bundle is not None else entry.parent
                relative = path.relative_to(base).as_posix()
                found[relative] = path
        return found

    def signature(path):
        value = path.lstat()
        return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)

    selected = entries()
    initial = {relative: signature(path) for relative, path in selected.items()}
    manifest, files, total = [], 0, 0
    for relative, path in sorted(selected.items()):
        mode = initial[relative][2]
        if stat.S_ISLNK(mode):
            try:
                target = path.resolve(strict=True)
            except (OSError, RuntimeError):
                raise ValueError("The Unity build contains an unresolved symbolic link.") from None
            if not target.is_relative_to(root):
                raise ValueError("The Unity build contains an external symbolic link.")
            manifest.append([relative, "symlink", target.relative_to(root).as_posix()])
        elif stat.S_ISDIR(mode):
            manifest.append([relative, "directory"])
        elif stat.S_ISREG(mode):
            digest, size = hashlib.sha256(), 0
            with path.open("rb") as source:
                while chunk := source.read(1_048_576):
                    digest.update(chunk)
                    size += len(chunk)
            manifest.append([relative, "file", mode & 0o111, size, digest.hexdigest()])
            files += 1
            total += size
        else:
            raise ValueError("The Unity build contains an unsupported special file.")
    after = entries()
    if set(after) != set(selected) or any(signature(after[name]) != initial[name] for name in selected):
        raise ValueError("The Unity build changed while it was being fingerprinted.")
    identity = json.dumps({"layout": layout, "entry": relative_entry, "manifest": manifest},
                          sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return {"schema": "unity-player-content-v1", "sha256": hashlib.sha256(identity).hexdigest(),
            "files": files, "total_bytes": total, "layout": layout}
