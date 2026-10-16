"""Actual build data changes must invalidate same-version study comparisons."""

import os
from pathlib import Path
import tempfile
import unittest

from autoplay_qa.native_build import fingerprint_unity_player


class NativeBuildTests(unittest.TestCase):
    def app(self, directory):
        bundle = Path(directory) / "Player.app"
        executable = bundle / "Contents/MacOS/Player"
        data = bundle / "Contents/Resources/Data/Managed/Game.dll"
        executable.parent.mkdir(parents=True)
        data.parent.mkdir(parents=True)
        executable.write_bytes(b"native player")
        data.write_bytes(b"actual game implementation")
        executable.chmod(0o755)
        return executable, data

    def test_same_version_player_data_rebuild_changes_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            executable, data = self.app(directory)
            first = fingerprint_unity_player(executable)
            self.assertEqual(first["files"], 2)
            data.write_bytes(b"different actual game implementation")
            self.assertNotEqual(first["sha256"], fingerprint_unity_player(executable)["sha256"])

    def test_timestamp_alone_is_not_a_build_change_and_no_local_paths_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            executable, data = self.app(directory)
            first = fingerprint_unity_player(executable)
            os.utime(data, (1, 1))
            self.assertEqual(first, fingerprint_unity_player(executable))
            self.assertNotIn(directory, str(first))

    def test_added_or_removed_runtime_file_changes_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            executable, data = self.app(directory)
            first = fingerprint_unity_player(executable)
            library = data.parent / "Plugin.dll"
            library.write_bytes(b"plugin")
            second = fingerprint_unity_player(executable)
            self.assertNotEqual(first["sha256"], second["sha256"])
            library.unlink()
            self.assertEqual(first, fingerprint_unity_player(executable))

    def test_internal_links_are_pinned_and_external_links_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            executable, data = self.app(directory)
            link = data.parent / "Current.dll"
            link.symlink_to(data.name)
            first = fingerprint_unity_player(executable)
            data.write_bytes(b"changed target")
            self.assertNotEqual(first["sha256"], fingerprint_unity_player(executable)["sha256"])
            link.unlink()
            outside = Path(directory) / "external.dll"
            outside.write_bytes(b"outside")
            link.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "external"):
                fingerprint_unity_player(executable)

    def test_standalone_layout_includes_data_and_engine_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "Player.x86_64"
            executable.write_bytes(b"native")
            (root / "Player_Data").mkdir()
            data = root / "Player_Data/game.data"
            data.write_bytes(b"data")
            engine = root / "UnityPlayer.so"
            engine.write_bytes(b"engine")
            first = fingerprint_unity_player(executable)
            self.assertEqual(first["files"], 3)
            engine.write_bytes(b"new engine")
            self.assertNotEqual(first["sha256"], fingerprint_unity_player(executable)["sha256"])


if __name__ == "__main__":
    unittest.main()
