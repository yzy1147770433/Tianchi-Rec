import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_pipeline


class PipelineValidationTest(unittest.TestCase):
    def test_raw_data_validation_lists_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            with patch.object(run_pipeline, 'DATA_DIR', data_dir):
                with self.assertRaisesRegex(FileNotFoundError, 'articles_emb.csv'):
                    run_pipeline.check_raw_data()

                for filename in run_pipeline.REQUIRED_RAW_FILES:
                    (data_dir / filename).touch()
                run_pipeline.check_raw_data()

    def test_recall_output_matches_recall_mode(self):
        output_dir = Path('artifacts') / 'test'
        self.assertEqual(
            run_pipeline.recall_output(output_dir, 'multi').name,
            'final_recall_items_dict.pkl',
        )
        self.assertEqual(
            run_pipeline.recall_output(output_dir, 'itemcf').name,
            'itemcf_recall_dict.pkl',
        )


if __name__ == '__main__':
    unittest.main()
