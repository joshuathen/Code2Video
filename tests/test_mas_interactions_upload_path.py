import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mas_interactions import _ascii_upload_path  # noqa: E402


class UploadPathTests(unittest.TestCase):
    def test_ascii_path_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video.mp4"
            source.write_bytes(b"video")
            with _ascii_upload_path(source) as upload_path:
                self.assertEqual(upload_path, source)

    def test_unicode_filename_uses_temporary_ascii_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video_π.mp4"
            source.write_bytes(b"video")
            with _ascii_upload_path(source) as upload_path:
                str(upload_path).encode("ascii")
                self.assertNotEqual(upload_path, source)
                self.assertTrue(upload_path.exists())
                self.assertEqual(upload_path.read_bytes(), source.read_bytes())
            self.assertTrue(source.exists())
            self.assertFalse(upload_path.exists())


if __name__ == "__main__":
    unittest.main()
