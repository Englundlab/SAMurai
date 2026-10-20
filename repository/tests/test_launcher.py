"""Check launcher device selection without starting inference or a GUI."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

class LauncherTests(unittest.TestCase):
    def test_device_selection(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('spheroid_launcher', root / 'run.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        import app.core  # Load dependencies before temporarily replacing the GUI module.
        config = str(root / 'app' / 'workflow_config.json')
        for requested, gpu_ok, expected in [('cpu', False, 'cpu'), ('cuda', False, 'cuda'), ('auto', False, 'cpu'), ('auto', True, 'cuda')]:
            with self.subTest(requested=requested, gpu_ok=gpu_ok):
                captured = []
                fake = types.ModuleType('app.app')
                fake.main = lambda: captured.extend(sys.argv)
                with patch.dict(sys.modules, {'app.app': fake}), patch.object(sys, 'argv', ['run.py', '--device', requested]), patch.object(launcher, 'gpu_works', return_value=gpu_ok) as probe:
                    launcher.main()
                    self.assertEqual(captured[1:], ['--config', config, '--device', expected])
                    self.assertEqual(probe.call_count, int(requested == 'auto'))

