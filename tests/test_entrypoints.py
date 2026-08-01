import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EntrypointTest(unittest.TestCase):
    def test_stage_wrappers_are_safe_to_import(self):
        with tempfile.TemporaryDirectory() as empty_data:
            env = os.environ.copy()
            env['TIANCHI_DATA_DIR'] = empty_data
            process = subprocess.run(
                [
                    sys.executable,
                    '-c',
                    'import Recall, tezhenggongcheng; '
                    'assert callable(Recall.main); '
                    'assert callable(tezhenggongcheng.main)',
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_package_cli_exposes_help_without_reading_data(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'src')
        process = subprocess.run(
            [sys.executable, '-m', 'tianchi_rec', '--help'],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('--mode', process.stdout)
        self.assertIn('--recall', process.stdout)


if __name__ == '__main__':
    unittest.main()
