"""Synthetic software fixtures only; these are not water-quality experiments."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from config import FEATURES, ABSTRACT_ARTIFACT_DIR
from future_work.core import FuturePredictor, labeled, validate_time, reject_overlap
from future_work.training import multiclass, timeseries, transfer, dann, GradientReversal


def fixture(seed=21, n=180):
    rng=np.random.default_rng(seed)
    frame=pd.DataFrame(rng.normal(size=(n,9)),columns=FEATURES)
    frame['Potability']=(frame.ph>0).astype(int)
    return frame


class FutureWorkIntegration(unittest.TestCase):
    def test_real_data_requirements_and_overlap_rejection(self):
        frame=fixture()
        with self.assertRaises(ValueError): labeled(frame,'Potability',multiclass=True)
        with self.assertRaises(ValueError): validate_time(frame)
        with self.assertRaises(ValueError): reject_overlap(frame,frame.iloc[:3])
        integers=pd.DataFrame(np.ones((2,9),dtype=int),columns=FEATURES)
        with self.assertRaises(ValueError): reject_overlap(integers,integers.astype(float))
        frame['timestamp']=pd.date_range('2020-01-01',periods=len(frame),freq='h')
        frame.loc[4,'timestamp']=frame.loc[3,'timestamp']
        with self.assertRaises(ValueError): validate_time(frame)

    def test_multiclass_training_and_saved_hybrid_reload(self):
        frame=fixture(); frame['WQI_class']=np.tile(['A','B','C'],60)
        with tempfile.TemporaryDirectory() as directory:
            report=multiclass(frame,Path(directory)/'run',epochs=1)
            restored=FuturePredictor(Path(directory)/'run')
            pred=restored.predict(frame.iloc[report['split_indices']['test']])
            saved=pd.read_csv(Path(directory)/'run/test_predictions.csv')
            np.testing.assert_allclose(pred.filter(like='probability_'),saved.filter(like='probability_'),atol=1e-6)
            self.assertEqual(restored.meta['classes'],['A','B','C'])
            self.assertEqual(len(report['test']),4)

    def test_chronological_lstm_windows_and_forecast_reload(self):
        frame=fixture(n=200); frame['timestamp']=pd.date_range('2020-01-01',periods=len(frame),freq='h')
        frame['WQI']=50+10*np.sin(np.arange(len(frame))*.1)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'
            report=timeseries(frame,out,lookback=4,horizon=2,epochs=1)
            tr,va,te=report['target_indices']
            self.assertLess(max(tr),min(va)); self.assertLess(max(va),min(te))
            self.assertGreaterEqual(min(va)-2-4+1,report['partition_bounds'][1][0])
            pred=FuturePredictor(out).predict(frame.iloc[-4:])
            self.assertTrue(np.isfinite(pred.predicted_WQI.iloc[0]))
            expected=pd.to_datetime(frame.timestamp.iloc[-1],utc=True)+pd.Timedelta(hours=2)
            self.assertEqual(pd.to_datetime(pred.forecast_timestamp.iloc[0]),expected)
            with self.assertRaises(ValueError): FuturePredictor(out).predict(frame.iloc[-3:])
            bad=frame.iloc[-4:].copy(); bad['timestamp']=pd.date_range('2020-01-01',periods=4,freq='2h')
            with self.assertRaises(ValueError): FuturePredictor(out).predict(bad)

    def test_transfer_frozen_preprocessing_and_reload(self):
        frame=fixture(seed=22)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'
            report=transfer(frame,ABSTRACT_ARTIFACT_DIR,out,epochs=1)
            self.assertIn('source_ann_zero_shot',report['test'])
            restored=FuturePredictor(out)
            pred=restored.predict(frame.iloc[report['split_indices']['test']])
            saved=pd.read_csv(out/'test_predictions.csv')
            np.testing.assert_allclose(pred.probability_Potable,saved.potable_probability,atol=1e-6)
            self.assertEqual((out/'preprocess.joblib').read_bytes(),(ABSTRACT_ARTIFACT_DIR/'preprocess.joblib').read_bytes())

    def test_gradient_reversal_and_serialization(self):
        x=tf.Variable([[1.,2.]])
        layer=GradientReversal(.7)
        with tf.GradientTape() as tape: value=tf.reduce_sum(layer(x))
        np.testing.assert_allclose(tape.gradient(value,x),[[-.7,-.7]])
        np.testing.assert_array_equal(layer(x),x)
        restored=tf.keras.layers.deserialize(tf.keras.layers.serialize(layer))
        self.assertEqual(restored.strength,.7)

    def test_dann_unlabeled_adaptation_and_heldout_target(self):
        source=fixture(seed=24); target=fixture(seed=25).drop(columns='Potability'); test=fixture(seed=26)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'
            report=dann(source,target,out,target_test=test,epochs=1)
            self.assertFalse(report['target_labels_used_for_training'])
            self.assertIn('target_test',report)
            pred=FuturePredictor(out).predict(test)
            saved=pd.read_csv(out/'target_test_predictions.csv')
            np.testing.assert_allclose(pred.probability_Potable,saved.potable_probability,atol=1e-6)
            with self.assertRaises(ValueError): dann(source,target,out,target_test=source,epochs=1)


if __name__=='__main__': unittest.main()
