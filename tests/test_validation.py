"""Regression cases found during the deep review."""
import io
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from config import FEATURES, ABSTRACT_ARTIFACT_DIR
from wqi.validation import read_csv, model_inputs, probabilities
from wqi.inference import Predictor


class ValidationRegression(unittest.TestCase):
    def test_raw_duplicate_and_empty_csv_headers(self):
        for text in ['ph,ph\n1,2\n', ',ph\n1,2\n', '']:
            with self.assertRaises(ValueError): read_csv(io.BytesIO(text.encode()))
        stream=io.BytesIO(b'\xef\xbb\xbfph,Hardness\n7,100\n')
        self.assertEqual(read_csv(stream).columns.tolist(), ['ph','Hardness'])
        self.assertEqual(stream.tell(),0)

    def test_float32_and_probability_boundaries(self):
        for values in [[1e300], [np.nan], [np.inf]]:
            with self.assertRaises(ValueError): model_inputs(values)
        for values in [[np.nan], [-.1], [1.1]]:
            with self.assertRaises(ValueError): probabilities(values)
        np.testing.assert_array_equal(probabilities([0,.5,1]),[0,.5,1])

    def test_extreme_measurements_never_become_nan_classification(self):
        predictor=Predictor(ABSTRACT_ARTIFACT_DIR)
        frame=pd.DataFrame(np.full((1,9),1e300),columns=FEATURES)
        with self.assertRaisesRegex(ValueError,'float32'): predictor.predict_frame(frame)
        frame=pd.DataFrame([[pd.NA]*9],columns=FEATURES,dtype='Float64')
        result=predictor.predict_frame(frame)
        self.assertTrue(np.isfinite(result.potable_probability).all())
        self.assertEqual(result.imputed_measurements.iloc[0],9)
        with patch('wqi.inference.probability',return_value=np.array([np.nan])):
            with self.assertRaises(ValueError): predictor.predict_frame(frame)


if __name__=='__main__': unittest.main()
