import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from mas import _resolve_belief_library_path  # noqa: E402


class BeliefLibraryPathResolutionTests(unittest.TestCase):
    def test_omitted_path_does_not_auto_discover_beliefs(self):
        self.assertIsNone(_resolve_belief_library_path(None))
        self.assertIsNone(_resolve_belief_library_path(""))

    def test_explicit_library_path_is_resolved(self):
        expected = Path(__file__).resolve()
        self.assertEqual(
            _resolve_belief_library_path(str(expected)),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
