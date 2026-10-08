"""Exercise real Streamlit pages and the trained model prediction form."""
import importlib.util
import unittest
from pathlib import Path
from config import ARTIFACT_DIR

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit not installed')
class AppIntegration(unittest.TestCase):
    def test_home_and_eda_render(self):
        from streamlit.testing.v1 import AppTest
        for name in ['app/app.py', 'app/pages/1_EDA.py']:
            with self.subTest(page=name):
                app = AppTest.from_file(str(ROOT / name)).run(timeout=30)
                self.assertEqual(len(app.exception), 0)

    @unittest.skipUnless((ARTIFACT_DIR / 'metadata.json').exists(), 'Train artifacts first')
    def test_prediction_and_experiment_pages(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app/pages/2_Prediction.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.number_input), 9)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].label, 'Model probability of potable class')
        experiments = AppTest.from_file(str(ROOT / 'app/pages/3_Experiments.py')).run(timeout=30)
        self.assertEqual(len(experiments.exception), 0)
        self.assertGreater(len(experiments.dataframe), 0)


if __name__ == '__main__':
    unittest.main()
