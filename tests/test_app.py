"""Exercise real Streamlit pages and the trained model prediction form."""
import importlib.util
import unittest
from pathlib import Path
from config import ARTIFACT_DIR, ABSTRACT_ARTIFACT_DIR

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit not installed')
class AppIntegration(unittest.TestCase):
    def test_home_and_eda_render(self):
        from streamlit.testing.v1 import AppTest
        for name in ['app/app.py', 'app/pages/1_EDA.py', 'app/pages/4_Future_Work.py']:
            with self.subTest(page=name):
                app = AppTest.from_file(str(ROOT / name)).run(timeout=30)
                self.assertEqual(len(app.exception), 0)

    def test_csv_upload_handlers(self):
        import io
        import pandas as pd
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        data=pd.read_csv(ROOT/'data/water_potability.csv').head(5).to_csv(index=False).encode()
        for payload, valid in [(data,True),(b'ph,ph\n1,2\n',False)]:
            with self.subTest(valid=valid), patch('streamlit.file_uploader',return_value=io.BytesIO(payload)):
                app=AppTest.from_file(str(ROOT/'app/pages/2_Prediction.py')).run(timeout=30)
                self.assertEqual(len(app.exception),0)
                if valid:
                    self.assertTrue(any(len(item.value)==5 for item in app.dataframe))
                else:
                    self.assertGreater(len(app.error),0)

    @unittest.skipUnless((ARTIFACT_DIR / 'metadata.json').exists(), 'Train artifacts first')
    def test_prediction_and_experiment_pages(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app/pages/2_Prediction.py')).run(timeout=30)
        # Both independent snapshots must load and support the same form.
        for mode in ['Paper reproduction attempt', 'Abstract implementation']:
            if mode == 'Abstract implementation' and not (ABSTRACT_ARTIFACT_DIR / 'metadata.json').exists():
                continue
            app.selectbox[0].select(mode).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.number_input), 9)
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.metric[0].label, 'Model probability of potable class')
        experiments = AppTest.from_file(str(ROOT / 'app/pages/3_Experiments.py')).run(timeout=30)
        for mode in ['Paper reproduction attempt', 'Abstract implementation']:
            if mode == 'Abstract implementation' and not (ABSTRACT_ARTIFACT_DIR / 'metadata.json').exists():
                continue
            experiments.selectbox[0].select(mode).run(timeout=30)
            self.assertEqual(len(experiments.exception), 0)
            self.assertGreater(len(experiments.dataframe), 0)


if __name__ == '__main__':
    unittest.main()
