"""Integration checks run when the training dependencies are installed."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from config import FEATURES

@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow not installed')
class NeuralIntegration(unittest.TestCase):
    def test_initialization_training_and_reload_predictions(self):
        from wqi.neural import build_ann, fit_ann, probability
        from tensorflow.keras.models import load_model
        weights=np.arange(1,10)/45
        m=build_ann(weights)
        self.assertEqual(m.input_shape,(None,9))
        self.assertEqual(m.output_shape,(None,1))
        np.testing.assert_allclose(m.get_layer('feature_layer').get_weights()[0][:,0],weights,rtol=1e-6)
        rng=np.random.default_rng(42); X=rng.uniform(size=(64,9)).astype('float32')
        y=(X[:,0]>.5).astype('float32')
        history=fit_ann(m,X[:48],y[:48],X[48:],y[48:],epochs=2)
        self.assertEqual(len(history['loss']),2)
        p=probability(m,X[48:]); self.assertTrue(((p>=0)&(p<=1)).all())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.keras'; m.save(path)
            restored=load_model(path,compile=False)
            np.testing.assert_allclose(p,probability(restored,X[48:]),atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec('shap') and importlib.util.find_spec('xgboost'), 'SHAP/XGBoost missing')
    def test_entire_training_artifact_inference_roundtrip(self):
        import importlib.util
        import json
        import joblib
        from wqi.core import load_data
        from wqi.inference import Predictor
        spec=importlib.util.spec_from_file_location('trainer',Path(__file__).resolve().parents[1]/'model/train_model.py')
        trainer=importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
        rng=np.random.default_rng(11)
        X=pd.DataFrame(rng.normal(size=(160,9)),columns=FEATURES)
        y=pd.Series((X.ph>0).astype(int))
        run=trainer.fit_experiment(X,y,42,2)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            joblib.dump(run['prep'],out/'preprocess.joblib')
            joblib.dump(run['combiner'],out/'combiner.joblib')
            run['ann'].save(out/'ann.keras'); run['xgb'].save_model(out/'xgboost.json')
            result=run['result']; fusion='stacking' if result['selected_fusion']=='stacking' else 'average'
            (out/'metadata.json').write_text(json.dumps({'schema':1,'features':FEATURES,'fusion':fusion,'alpha':result['alpha']}))
            p,_,_=Predictor(out).predict(run['Xt'])
            np.testing.assert_allclose(p,run['predictions']['hybrid_selected'],atol=1e-6)
        self.assertEqual(set(run['result']['test']),set(run['predictions']))
        self.assertIn('ann_shap',run['result']['test'])

if __name__=='__main__': unittest.main()
