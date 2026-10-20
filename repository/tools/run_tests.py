"""Run the repository tests, including with isolated portable Python runtimes."""
from pathlib import Path
import sys
import unittest
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(str(root / 'tests'), top_level_dir=str(root))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
